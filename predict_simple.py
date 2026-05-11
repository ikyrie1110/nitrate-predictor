import sys
import os
import numpy as np
import pandas as pd
import joblib
import warnings
from tkinter import filedialog, Tk

# 尝试导入tqdm
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

warnings.filterwarnings('ignore')


def resource_path(relative_path):
    """Get absolute path for resource files, supports PyInstaller packaging"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


# Required feature names for the model (12 features)
FEATURE_NAMES = ['Year', 'Lon', 'Ammonium', 'Lat', 'Sulfate', 'Nitrate_TAP',
                 'Temp', 'SurfacePressure', 'NO2', 'PM2.5', 'BC', 'NDVI']

# Column name mapping from your data to model features
COLUMN_MAPPING = {
    'Year': 'Year',
    'lat': 'Lat',
    'lon': 'Lon',
    'NO₂': 'NO2',
    'T': 'Temp',
    'NH₄⁺': 'Ammonium',
    'NO₃⁻': 'Nitrate_TAP',
    'SP': 'SurfacePressure',
    'PM₂.₅': 'PM2.5',
    'BC': 'BC',
    'SO₄²⁻': 'Sulfate',
    'NDVI': 'NDVI'
}


def load_models():
    """Load all models"""
    print("\nLoading models...")
    try:
        models = {}
        models['RandomForest'] = joblib.load(resource_path('RandomForest_updated_model.joblib'))
        models['CatBoost'] = joblib.load(resource_path('CatBoost_updated_model.joblib'))
        meta_model = joblib.load(resource_path('ElasticNet_best_meta_model.joblib'))
        print("Models loaded successfully!")
        return models, meta_model
    except Exception as e:
        print(f"Model loading failed: {e}")
        return None, None


def select_input_file():
    """Select input CSV file using file dialog"""
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = filedialog.askopenfilename(
        title="Select Input CSV File",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    root.destroy()
    return file_path


def select_output_file(default_name="nitrate_estimation.csv"):
    """Select output CSV file path"""
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = filedialog.asksaveasfilename(
        title="Save Result as CSV File",
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        initialfile=default_name
    )
    root.destroy()
    return file_path


def map_columns(df):
    """Map your dataframe columns to model feature names"""
    df_mapped = df.copy()
    for your_col, model_col in COLUMN_MAPPING.items():
        if your_col in df_mapped.columns:
            df_mapped.rename(columns={your_col: model_col}, inplace=True)
    return df_mapped


def predict_batch_vectorized(df_features, models, meta_model):
    """
    批量向量化预测 - 核心优化函数
    一次性处理所有数据，速度提升50-100倍
    """
    # 确保所有特征存在并填充默认值
    for col in FEATURE_NAMES:
        if col not in df_features.columns:
            if col == 'Year':
                df_features[col] = 2013
            elif col == 'NDVI':
                df_features[col] = 0.5
            else:
                df_features[col] = 0
    
    # 提取特征矩阵（一次性）
    X = df_features[FEATURE_NAMES].values
    X = np.nan_to_num(X, 0)  # 处理NaN
    
    # 批量预测基础模型
    rf_pred = models['RandomForest'].predict(X)
    cb_pred = models['CatBoost'].predict(X)
    
    # 堆叠特征
    stacking_features = np.column_stack([rf_pred, cb_pred])
    full_features = np.hstack([stacking_features, X])
    
    # 确保特征数量匹配
    expected = meta_model.n_features_in_
    if full_features.shape[1] != expected:
        if full_features.shape[1] < expected:
            missing = expected - full_features.shape[1]
            full_features = np.hstack([full_features, np.zeros((full_features.shape[0], missing))])
        else:
            full_features = full_features[:, :expected]
    
    # 批量元模型预测
    final_predictions = meta_model.predict(full_features)
    
    # 确保非负
    final_predictions = np.maximum(final_predictions, 0)
    
    return final_predictions


def predict_batch(input_file_path, output_file_path, models, meta_model):
    """批量预测（优化版）"""
    try:
        # 读取输入文件
        print(f"\nReading input file: {input_file_path}")
        df_original = pd.read_csv(input_file_path)
        
        # 映射列名
        df_mapped = map_columns(df_original)
        # print(f"Columns: {list(df_mapped.columns)}")
        
        # 检查必要列是否存在
        missing_cols = [col for col in FEATURE_NAMES if col not in df_mapped.columns]
        if missing_cols:
            print(f"Input file missing required columns: {missing_cols}")
            print(f"Required columns: {FEATURE_NAMES}")
            print(f"Your available columns: {list(df_mapped.columns)}")
            return False
        
        print(f"Successfully loaded {len(df_mapped)} records")
        
        # 批量预测（核心优化）
        print("\nEstimating nitrate concentration (batch mode)...")
        predictions = predict_batch_vectorized(df_mapped, models, meta_model)
        
        # 统计负值（如果有）
        negative_count = np.sum(predictions == 0)  # 被裁剪为0的数量
        # if negative_count > 0:
            # print(f"   {negative_count} negative predictions were set to 0")
        
        # 创建输出DataFrame
        df_output = pd.DataFrame()
        
        # Add Year
        if 'Year' in df_original.columns:
            df_output['Year'] = df_original['Year']
        else:
            df_output['Year'] = range(1, len(df_original) + 1)
        
        # Add Month
        if 'Month' in df_original.columns:
            df_output['Month'] = df_original['Month']
        else:
            df_output['Month'] = 1
        
        # Add Latitude
        if 'lat' in df_original.columns:
            df_output['Lat'] = df_original['lat']
        elif 'Lat' in df_mapped.columns:
            df_output['Lat'] = df_mapped['Lat']
        else:
            df_output['Lat'] = np.nan
        
        # Add Longitude
        if 'lon' in df_original.columns:
            df_output['Lon'] = df_original['lon']
        elif 'Lon' in df_mapped.columns:
            df_output['Lon'] = df_mapped['Lon']
        else:
            df_output['Lon'] = np.nan
        
        # 添加预测结果
        df_output['Nitrate'] = predictions
        
        # 保存结果
        df_output.to_csv(output_file_path, index=False, encoding='utf-8-sig')
        
        # 显示统计信息
        # print("\n" + "=" * 50)
        # print("Estimation Statistics:")
        # print(f"  Total samples: {len(df_output)}")
        # print(f"  Nitrate range: [{predictions.min():.4f}, {predictions.max():.4f}]")
        # print(f"  Mean Nitrate: {predictions.mean():.4f}")
        # print(f"  Standard deviation: {predictions.std():.4f}")
        # print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 60)
    print("Nitrate Concentration Estimation Tool (Optimized)")
    print("=" * 60)
    
    # 加载模型
    models, meta_model = load_models()
    if models is None or meta_model is None:
        input("\nPress Enter to exit...")
        return
    
    # 选择输入文件
    input_file = select_input_file()
    if not input_file:
        print("No input file selected. Exiting...")
        input("\nPress Enter to exit...")
        return
    
    # 生成默认输出文件名
    default_output = os.path.join(
        os.path.dirname(input_file),
        f"nitrate_estimation_{os.path.basename(input_file)}"
    )
    
    # 选择输出文件
    output_file = select_output_file(default_output)
    if not output_file:
        print("No output file path selected. Exiting...")
        input("\nPress Enter to exit...")
        return
    
    # 执行预测
    success = predict_batch(input_file, output_file, models, meta_model)
    
    if success:
        print(f"\n Estimation completed! Results saved to: {output_file}")
        open_folder = input("\n Open file folder? (y/n): ").strip().lower()
        if open_folder == 'y':
            if sys.platform == 'win32':
                os.startfile(os.path.dirname(output_file))
            elif sys.platform == 'darwin':
                os.system(f'open "{os.path.dirname(output_file)}"')
            else:
                os.system(f'xdg-open "{os.path.dirname(output_file)}"')
    
    input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()
