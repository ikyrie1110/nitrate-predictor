import sys
import os
import numpy as np
import pandas as pd
import joblib
import warnings
from tkinter import filedialog, Tk
from tkinter import messagebox

# 尝试导入tqdm，如果没有则安装
try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # print("Tip: Install 'tqdm' for better progress bar: pip install tqdm")

warnings.filterwarnings('ignore')


def resource_path(relative_path):
    """Get absolute path for resource files, supports PyInstaller packaging"""
    if hasattr(sys, '_MEIPASS'):
        # Temporary directory after packaging
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


# Required feature names for the model (12 features)
FEATURE_NAMES = ['Year', 'Lon', 'Ammonium', 'Lat', 'Sulfate', 'Nitrate_TAP',
                 'Temp', 'SurfacePressure', 'NO2', 'PM2.5', 'BC', 'NDVI']

# Column name mapping from your data to model features
COLUMN_MAPPING = {
    'Year': 'Year',  # Your 'Year' -> model 'Year'
    'lat': 'Lat',  # Your 'lat' -> model 'Lat'
    'lon': 'Lon',  # Your 'lon' -> model 'Lon'
    'NO₂': 'NO2',  # Your 'NO₂' -> model 'NO2'
    'T': 'Temp',  # Your 'T' -> model 'Temp'
    'NH₄⁺': 'Ammonium',  # Your 'NH₄⁺' -> model 'Ammonium'
    'NO₃⁻': 'Nitrate_TAP',  # Your 'NO₃⁻' -> model 'Nitrate_TAP'
    'SP': 'SurfacePressure',  # Your 'SP' -> model 'SurfacePressure'
    'PM₂.₅': 'PM2.5',  # Your 'PM₂.₅' -> model 'PM2.5'
    'BC': 'BC',  # Your 'BC' -> model 'BC'
    'SO₄²⁻': 'Sulfate',  # Your 'SO₄²⁻' -> model 'Sulfate'
    'NDVI': 'NDVI'  # Your 'NDVI' -> model 'NDVI'
}


def load_models():
    """Load all models"""
    print("\nLoading models...")
    try:
        # Load base models
        models = {}
        models['RandomForest'] = joblib.load(resource_path('RandomForest_updated_model.joblib'))
        models['CatBoost'] = joblib.load(resource_path('CatBoost_updated_model.joblib'))

        # Load meta model
        meta_model = joblib.load(resource_path('ElasticNet_best_meta_model.joblib'))

        print("✅ Models loaded successfully!")
        return models, meta_model
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return None, None


def select_input_file():
    """Select input CSV file using file dialog"""
    root = Tk()
    root.withdraw()  # Hide main window
    root.attributes('-topmost', True)  # Bring dialog to front

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

    # Rename columns according to mapping
    for your_col, model_col in COLUMN_MAPPING.items():
        if your_col in df_mapped.columns:
            df_mapped.rename(columns={your_col: model_col}, inplace=True)

    return df_mapped


def create_progress_bar(total, desc="Estimating"):
    """创建进度条"""
    if TQDM_AVAILABLE:
        return tqdm(total=total, desc=desc, unit='samples',
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    else:
        # 简单的文本进度条
        return SimpleProgressBar(total, desc)


class SimpleProgressBar:
    """简单的文本进度条（当tqdm不可用时）"""

    def __init__(self, total, desc="Progress"):
        self.total = total
        self.desc = desc
        self.current = 0
        self.last_percent = 0

    def update(self, n=1):
        self.current += n
        percent = int(100 * self.current / self.total)
        if percent > self.last_percent:
            self.last_percent = percent
            # 每10%显示一次
            if percent % 10 == 0:
                print(f"  {self.desc}: {percent}% ({self.current}/{self.total})")

    def close(self):
        print(f"  {self.desc}: 100% ({self.total}/{self.total}) - Complete!")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def predict_batch(input_file_path, output_file_path, models, meta_model):
    """Batch prediction and save results"""
    try:
        # Read input CSV
        print(f"\n📖 Reading input file: {input_file_path}")
        df_original = pd.read_csv(input_file_path)

        # print(f"📋 Original columns: {list(df_original.columns)}")

        # Map column names for feature extraction
        df_mapped = map_columns(df_original)
        print(f"columns: {list(df_mapped.columns)}")

        # Check if necessary feature columns exist after mapping
        missing_cols = [col for col in FEATURE_NAMES if col not in df_mapped.columns]
        if missing_cols:
            print(f"❌ Input file missing required columns: {missing_cols}")
            print(f"Required columns: {FEATURE_NAMES}")
            print(f"Your available columns: {list(df_mapped.columns)}")
            return False

        print(f"✅ Successfully loaded {len(df_mapped)} records")

        # Store predictions
        predictions = []

        # Predict row by row using the 12 features with progress bar
        print("\n🔮 Estimating nitrate concentration...")

        # 创建进度条
        total_samples = len(df_mapped)
        if TQDM_AVAILABLE:
            progress_bar = tqdm(total=total_samples, desc="  Progress",
                                unit='samples', ncols=80,
                                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        else:
            progress_bar = SimpleProgressBar(total_samples, "  Progress")

        for idx, row in df_mapped.iterrows():
            # Extract the 12 features
            features = row[FEATURE_NAMES].values.reshape(1, -1)

            # Base model predictions
            base_predictions = []
            for name, model in models.items():
                pred = model.predict(features)[0]
                base_predictions.append(pred)

            # Build stacking features
            stacking_features = np.array(base_predictions).reshape(1, -1)
            full_features = np.hstack([stacking_features, features])

            # Handle feature count mismatch
            expected = meta_model.n_features_in_
            if full_features.shape[1] != expected:
                if full_features.shape[1] < expected:
                    missing = expected - full_features.shape[1]
                    full_features = np.hstack([full_features, np.zeros((1, missing))])
                else:
                    full_features = full_features[:, :expected]

            # Meta model prediction
            final_prediction = meta_model.predict(full_features)[0]
            predictions.append(final_prediction)

            # Update progress bar
            progress_bar.update(1)

        progress_bar.close()

        # Create output DataFrame with Year, Month, Lat, Lon, Nitrate
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
            print("⚠️  Warning: No Month column found, using 1 as default")
            df_output['Month'] = 1

        # Add Latitude (Lat)
        if 'lat' in df_original.columns:
            df_output['Lat'] = df_original['lat']
        elif 'Lat' in df_mapped.columns:
            df_output['Lat'] = df_mapped['Lat']
        else:
            print("⚠️  Warning: No latitude column found")
            df_output['Lat'] = np.nan

        # Add Longitude (Lon)
        if 'lon' in df_original.columns:
            df_output['Lon'] = df_original['lon']
        elif 'Lon' in df_mapped.columns:
            df_output['Lon'] = df_mapped['Lon']
        else:
            print("⚠️  Warning: No longitude column found")
            df_output['Lon'] = np.nan

        # Add estimation result as Nitrate
        df_output['Nitrate'] = predictions

        # Save results
        print(f"\n💾 Saving results to: {output_file_path}")
        df_output.to_csv(output_file_path, index=False, encoding='utf-8-sig')

        # Display statistics
        print("\n" + "=" * 50)
        print("📊 Estimation Statistics:")
        print(f"  📍 Total samples: {len(df_output)}")
        print(f"  📈 Nitrate range: [{min(predictions):.4f}, {max(predictions):.4f}]")
        print(f"  📊 Mean Nitrate: {np.mean(predictions):.4f}")
        print(f"  📉 Standard deviation: {np.std(predictions):.4f}")
        print("=" * 50)

        # Show first few rows of output
        print("\n📄 First 5 rows of output:")
        print(df_output.head())

        return True

    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 60)
    print("🎯 Nitrate Concentration Estimation Tool")
    print("=" * 60)

    # # 显示进度条状态
    # if TQDM_AVAILABLE:
    #     # print("✨ Progress bar: Enhanced (tqdm)")
    # else:
    #     # print("📊 Progress bar: Simple mode (install tqdm for better display)")
    #     print("   Run: pip install tqdm")

    # Load models
    models, meta_model = load_models()
    if models is None or meta_model is None:
        input("\nPress Enter to exit...")
        return

    # Select input file
    input_file = select_input_file()
    if not input_file:
        print("No input file selected. Exiting...")
        input("\nPress Enter to exit...")
        return

    # Generate default output filename
    default_output = os.path.join(
        os.path.dirname(input_file),
        f"nitrate_estimation_{os.path.basename(input_file)}"
    )

    # Select output file
    output_file = select_output_file(default_output)
    if not output_file:
        print("No output file path selected. Exiting...")
        input("\nPress Enter to exit...")
        return

    # Perform estimation
    success = predict_batch(input_file, output_file, models, meta_model)

    if success:
        print(f"\n✅ Estimation completed! Results saved to: {output_file}")
        # Ask whether to open the file folder
        open_folder = input("\n📂 Open file folder? (y/n): ").strip().lower()
        if open_folder == 'y':
            if sys.platform == 'win32':
                os.startfile(os.path.dirname(output_file))
            elif sys.platform == 'darwin':  # macOS
                os.system(f'open "{os.path.dirname(output_file)}"')
            else:  # Linux
                os.system(f'xdg-open "{os.path.dirname(output_file)}"')

    input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()
