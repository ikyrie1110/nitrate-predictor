import sys
import numpy as np
import joblib
import warnings

warnings.filterwarnings('ignore')

FEATURE_NAMES = ['Year', 'lon', 'NH₄⁺', 'lat', 'SO₄²⁻', 'TAP_NO₃⁻',
                 'T', 'SP', 'NO₂', 'PM₂.₅', 'BC', 'NDVI']


def main():
    print("\n" + "=" * 50)
    print("硝酸盐浓度估算")
    print("=" * 50)

    # 加载模型
    print("\n正在加载模型...")
    try:
        # 加载基模型
        models = {}
        models['RandomForest'] = joblib.load('RandomForest_updated_model.joblib')
        models['CatBoost'] = joblib.load('CatBoost_updated_model.joblib')

        # 尝试加载其他模型（如果有）
        for name in ['GradientBoosting', 'XGBoost']:
            try:
                models[name] = joblib.load(f'{name}_updated_model.joblib')
            except:
                pass

        # 加载元模型
        meta_model = joblib.load('ElasticNet_best_meta_model.joblib')
    except Exception as e:
        print(f"模型加载失败: {e}")
        input("按回车键退出...")
        return

    print("\n" + "-" * 50)
    print("输入说明:")
    print("  请一次性输入12个数值，用空格分隔")
    print(f"  顺序: {' -> '.join(FEATURE_NAMES)}")
    print("  示例: 2020 116.4 12.5 39.9 15.2 8.3 15.6 1013 25.1 45.3 2.1 0.65")
    print("  输入 'quit' 退出程序")
    print("-" * 50)

    while True:
        print()
        user_input = input("请输入12个数值: ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("退出程序")
            break

        # 处理输入
        values = user_input.split()
        if len(values) != 12:
            print(f"❌ 需要12个数值，您输入了{len(values)}个")
            continue

        try:
            # 转换为 numpy 数组（不用 pandas）
            features = np.array([float(v) for v in values]).reshape(1, -1)

            # 基模型预测
            base_predictions = []
            for name, model in models.items():
                pred = model.predict(features)[0]
                base_predictions.append(pred)

            # 构建堆叠特征
            stacking_features = np.array(base_predictions).reshape(1, -1)
            full_features = np.hstack([stacking_features, features])

            # 处理特征数量不匹配
            expected = meta_model.n_features_in_
            if full_features.shape[1] != expected:
                if full_features.shape[1] < expected:
                    missing = expected - full_features.shape[1]
                    full_features = np.hstack([full_features, np.zeros((1, missing))])
                else:
                    full_features = full_features[:, :expected]

            # 元模型预测
            final_prediction = meta_model.predict(full_features)[0]

            print("-" * 40)
            print(f"🎯 最终估算结果: {final_prediction:.4f}")
            print("=" * 40)

        except ValueError:
            print("❌ 输入错误: 请确保输入的是数字")
        except Exception as e:
            print(f"❌ 预测失败: {e}")


if __name__ == "__main__":
    main()