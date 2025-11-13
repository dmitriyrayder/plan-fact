import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
from dateutil import parser
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

st.set_page_config(
    page_title="План/Факт Продаж Оптика",
    page_icon="👓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# ЗАГРУЗКА ДАННЫХ
# ============================================================================

# Определение необходимых колонок
REQUIRED_FACT_COLUMNS = ['Magazin', 'Datasales', 'Segment', 'Price', 'Qty', 'Sum']
REQUIRED_PLAN_COLUMNS = ['Magazin', 'Segment', 'Month', 'Revenue_Plan', 'Units_Plan']


def validate_columns(df, required_columns, data_type):
    """Проверка наличия всех необходимых колонок в DataFrame"""
    if df is None or df.empty:
        st.error(f"❌ {data_type}: данные пустые")
        return False

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        st.error(f"❌ {data_type}: отсутствуют обязательные колонки: {', '.join(missing_columns)}")
        st.info(f"📋 Найденные колонки: {', '.join(df.columns.tolist())}")
        st.info(f"📋 Ожидаемые колонки: {', '.join(required_columns)}")
        return False

    # Дополнительная валидация для числовых полей
    if data_type == "Факт":
        numeric_columns = ['Price', 'Qty', 'Sum']
        for col in numeric_columns:
            if col in df.columns:
                # Пробуем конвертировать в числовой тип
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    null_count = df[col].isna().sum()
                    if null_count > 0:
                        st.warning(f"⚠️ {data_type}: колонка '{col}' содержит {null_count} нечисловых значений, они будут заменены на 0")
                        df[col] = df[col].fillna(0)

                    # Проверка на отрицательные значения
                    negative_count = (df[col] < 0).sum()
                    if negative_count > 0:
                        st.warning(f"⚠️ {data_type}: колонка '{col}' содержит {negative_count} отрицательных значений, они будут заменены на 0")
                        df[col] = df[col].clip(lower=0)
                except Exception as e:
                    st.error(f"❌ {data_type}: ошибка преобразования колонки '{col}' в числовой формат: {str(e)}")
                    return False

        # Проверка математической консистентности: Sum должно равняться Price * Qty
        if all(col in df.columns for col in ['Price', 'Qty', 'Sum']):
            df['Expected_Sum'] = df['Price'] * df['Qty']
            # Допускаем погрешность 1% из-за округлений
            tolerance = 0.01
            df['Sum_Diff'] = abs(df['Sum'] - df['Expected_Sum'])
            # Безопасное деление с защитой от деления на ноль
            df['Sum_Diff_Pct'] = np.where(
                df['Expected_Sum'] != 0,
                df['Sum_Diff'] / df['Expected_Sum'],
                0
            )
            inconsistent_rows = (df['Sum_Diff_Pct'] > tolerance).sum()

            if inconsistent_rows > 0:
                st.warning(f"⚠️ {data_type}: обнаружено {inconsistent_rows} записей где Sum ≠ Price × Qty (с погрешностью > {tolerance*100}%)")
                st.info("💡 Совет: проверьте правильность расчета суммы в исходных данных")

            # Удаляем вспомогательные колонки
            df.drop(['Expected_Sum', 'Sum_Diff', 'Sum_Diff_Pct'], axis=1, inplace=True)

    if data_type == "План":
        numeric_columns = ['Revenue_Plan', 'Units_Plan']
        for col in numeric_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    null_count = df[col].isna().sum()
                    if null_count > 0:
                        st.warning(f"⚠️ {data_type}: колонка '{col}' содержит {null_count} нечисловых значений, они будут заменены на 0")
                        df[col] = df[col].fillna(0)

                    # Проверка на отрицательные значения
                    negative_count = (df[col] < 0).sum()
                    if negative_count > 0:
                        st.warning(f"⚠️ {data_type}: колонка '{col}' содержит {negative_count} отрицательных значений, они будут заменены на 0")
                        df[col] = df[col].clip(lower=0)
                except Exception as e:
                    st.error(f"❌ {data_type}: ошибка преобразования колонки '{col}' в числовой формат: {str(e)}")
                    return False

    return True


def parse_sheets_url(url):
    """Извлечение spreadsheet_id и gid из URL"""
    try:
        # Извлекаем spreadsheet_id
        if '/d/' in url:
            spreadsheet_id = url.split('/d/')[1].split('/')[0]
        else:
            spreadsheet_id = url

        # Извлекаем gid
        gid = None
        if '#gid=' in url:
            gid = url.split('#gid=')[1].split('&')[0]
        elif 'gid=' in url:
            gid = url.split('gid=')[1].split('&')[0]

        return spreadsheet_id, gid
    except BaseException:
        return None, None


@st.cache_data(ttl=600)
def load_data_from_sheets(plan_url, fact_url):
    """Загрузка из Google Sheets (публичный доступ)"""
    try:
        # Парсим ссылки
        plan_id, plan_gid = parse_sheets_url(plan_url)
        fact_id, fact_gid = parse_sheets_url(fact_url)

        if not plan_id or not plan_gid:
            st.error("❌ Некорректная ссылка на План")
            return None, None

        if not fact_id or not fact_gid:
            st.error("❌ Некорректная ссылка на Факт")
            return None, None

        # Формируем URLs для экспорта
        plan_export = f'https://docs.google.com/spreadsheets/d/{plan_id}/export?format=csv&gid={plan_gid}'
        fact_export = f'https://docs.google.com/spreadsheets/d/{fact_id}/export?format=csv&gid={fact_gid}'

        # Загрузка
        df_plan = pd.read_csv(plan_export)
        df_fact = pd.read_csv(fact_export)

        # Валидация колонок
        if not validate_columns(df_fact, REQUIRED_FACT_COLUMNS, "Факт"):
            return None, None

        if not validate_columns(df_plan, REQUIRED_PLAN_COLUMNS, "План"):
            return None, None

        st.success("✅ Данные успешно загружены и проверены")
        return df_fact, df_plan

    except Exception as e:
        st.error(f"❌ Ошибка загрузки: {str(e)}")
        st.info("💡 Проверь: таблица публична, ссылки правильные, колонки соответствуют требуемым")
        return None, None


def parse_dates_flexible(date_series):
    """
    Гибкий парсинг дат с поддержкой множественных форматов
    Поддерживаемые форматы:
    - YYYY-MM-DD (ISO 8601)
    - DD.MM.YYYY
    - DD/MM/YYYY
    - MM/DD/YYYY
    - YYYY/MM/DD
    - DD-MM-YYYY
    - Excel serial dates (числа)
    """
    parsed_dates = []
    errors = []

    for idx, date_val in enumerate(date_series):
        if pd.isna(date_val):
            parsed_dates.append(pd.NaT)
            continue

        try:
            # Если это уже datetime
            if isinstance(date_val, (pd.Timestamp, datetime)):
                parsed_dates.append(pd.Timestamp(date_val))
                continue

            # Если это число (Excel serial date)
            if isinstance(date_val, (int, float)):
                # Excel начинает считать с 1900-01-01
                try:
                    parsed_date = pd.Timestamp('1899-12-30') + pd.Timedelta(days=date_val)
                    parsed_dates.append(parsed_date)
                    continue
                except:
                    pass

            # Пробуем стандартные форматы
            date_str = str(date_val).strip()

            # Список форматов для проверки
            date_formats = [
                '%Y-%m-%d',      # 2025-01-15
                '%d.%m.%Y',      # 15.01.2025
                '%d/%m/%Y',      # 15/01/2025
                '%m/%d/%Y',      # 01/15/2025
                '%Y/%m/%d',      # 2025/01/15
                '%d-%m-%Y',      # 15-01-2025
                '%Y%m%d',        # 20250115
                '%d.%m.%y',      # 15.01.25
                '%d/%m/%y',      # 15/01/25
            ]

            parsed = None
            for fmt in date_formats:
                try:
                    parsed = datetime.strptime(date_str, fmt)
                    break
                except:
                    continue

            # Если не получилось стандартными форматами, пробуем dateutil
            if parsed is None:
                try:
                    parsed = parser.parse(date_str, dayfirst=True)
                except:
                    errors.append((idx, date_val))
                    parsed_dates.append(pd.NaT)
                    continue

            parsed_dates.append(pd.Timestamp(parsed))

        except Exception as e:
            errors.append((idx, date_val))
            parsed_dates.append(pd.NaT)

    return pd.Series(parsed_dates), errors


def load_data_from_excel(fact_file, plan_file):
    """Загрузка данных из Excel файлов"""
    try:
        # Загрузка файла Факт
        if fact_file is not None:
            # Определяем тип файла
            file_ext = fact_file.name.split('.')[-1].lower()

            if file_ext in ['xlsx', 'xls']:
                df_fact = pd.read_excel(fact_file, engine='openpyxl' if file_ext == 'xlsx' else None)
            elif file_ext == 'csv':
                # Пробуем определить разделитель
                df_fact = pd.read_csv(fact_file, encoding='utf-8-sig', sep=None, engine='python')
            else:
                st.error(f"❌ Неподдерживаемый формат файла Факт: {file_ext}")
                return None, None

            # Валидация колонок
            if not validate_columns(df_fact, REQUIRED_FACT_COLUMNS, "Факт"):
                return None, None

        else:
            st.error("❌ Файл Факт не загружен")
            return None, None

        # Загрузка файла План
        if plan_file is not None:
            file_ext = plan_file.name.split('.')[-1].lower()

            if file_ext in ['xlsx', 'xls']:
                df_plan = pd.read_excel(plan_file, engine='openpyxl' if file_ext == 'xlsx' else None)
            elif file_ext == 'csv':
                df_plan = pd.read_csv(plan_file, encoding='utf-8-sig', sep=None, engine='python')
            else:
                st.error(f"❌ Неподдерживаемый формат файла План: {file_ext}")
                return None, None

            # Валидация колонок
            if not validate_columns(df_plan, REQUIRED_PLAN_COLUMNS, "План"):
                return None, None

        else:
            st.error("❌ Файл План не загружен")
            return None, None

        st.success("✅ Файлы успешно загружены и проверены")
        return df_fact, df_plan

    except Exception as e:
        st.error(f"❌ Ошибка загрузки файлов: {str(e)}")
        st.info("💡 Проверьте: формат файлов корректный, колонки соответствуют требуемым")
        return None, None


@st.cache_data
def generate_demo_data():
    """Генерация демо данных"""
    np.random.seed(42)

    # Магазины
    stores = [f"Салон_{i}" for i in range(1, 71)]
    segments = ['Premium', 'Medium', 'Economy', 'Sun']
    months = pd.date_range('2025-01-01', '2025-03-31', freq='D')

    # Факт продаж
    fact_records = []
    for store in stores[:10]:  # Для демо только 10 магазинов
        for day in months:
            for segment in segments:
                n_sales = np.random.randint(5, 20)
                for _ in range(n_sales):
                    price = {
                        'Premium': np.random.uniform(15000, 30000),
                        'Medium': np.random.uniform(8000, 15000),
                        'Economy': np.random.uniform(3000, 8000),
                        'Sun': np.random.uniform(5000, 12000)
                    }[segment]

                    qty = 1
                    fact_records.append({
                        'Magazin': store,
                        'Datasales': day.strftime('%Y-%m-%d'),
                        'Art': f'ART{np.random.randint(1000, 9999)}',
                        'Describe': f'Оправа {segment}',
                        'Model': f'Model_{np.random.randint(1, 50)}',
                        'Segment': segment,
                        'Price': round(price, 2),
                        'Qty': qty,
                        'Sum': round(price * qty, 2)
                    })

    df_fact = pd.DataFrame(fact_records)

    # План продаж
    plan_records = []
    for store in stores[:10]:
        for month in ['2025-01', '2025-02', '2025-03']:
            for segment in segments:
                base_revenue = {
                    'Premium': 800000,
                    'Medium': 600000,
                    'Economy': 400000,
                    'Sun': 350000
                }[segment]

                revenue_plan = base_revenue * np.random.uniform(0.8, 1.2)
                units_plan = int(revenue_plan / (base_revenue / 150))

                plan_records.append({
                    'Magazin': store,
                    'Segment': segment,
                    'Month': month,
                    'Revenue_Plan': round(revenue_plan, 2),
                    'Units_Plan': units_plan
                })

    df_plan = pd.DataFrame(plan_records)

    return df_fact, df_plan

# Обработка данных


def prepare_data(df_fact, df_plan):
    """Подготовка данных для анализа"""

    # Валидация входных данных
    if df_fact is None or df_fact.empty:
        st.error("❌ Данные факта пустые или отсутствуют")
        return None, None

    if df_plan is None or df_plan.empty:
        st.error("❌ Данные плана пустые или отсутствуют")
        return None, None

    # Преобразование дат с поддержкой множественных форматов
    # Используем гибкую функцию парсинга дат
    df_fact['Datasales'], date_errors = parse_dates_flexible(df_fact['Datasales'])

    # Проверка на некорректные даты
    if date_errors:
        st.warning(f"⚠️ Обнаружено {len(date_errors)} записей с некорректными датами, они будут пропущены")
        if len(date_errors) <= 10:
            st.info(f"Примеры некорректных дат: {[err[1] for err in date_errors[:5]]}")

    invalid_dates = df_fact['Datasales'].isna().sum()
    if invalid_dates > 0:
        df_fact = df_fact.dropna(subset=['Datasales'])

    df_fact['Month'] = df_fact['Datasales'].dt.to_period('M').astype(str)
    df_fact['Week'] = df_fact['Datasales'].dt.to_period('W').astype(str)

    # Агрегация факта по магазин × сегмент × месяц
    fact_agg = df_fact.groupby(['Magazin', 'Segment', 'Month']).agg({
        'Sum': 'sum',
        'Qty': 'sum'
    }).reset_index()
    fact_agg.columns = [
        'Magazin',
        'Segment',
        'Month',
        'Revenue_Fact',
        'Units_Fact']

    # Объединение план и факт
    df_merged = pd.merge(
        df_plan,
        fact_agg,
        on=['Magazin', 'Segment', 'Month'],
        how='left'
    )

    # Заполнение NaN нулями
    df_merged['Revenue_Fact'] = df_merged['Revenue_Fact'].fillna(0)
    df_merged['Units_Fact'] = df_merged['Units_Fact'].fillna(0)

    # Расчет отклонений с защитой от деления на ноль
    df_merged['Revenue_Diff'] = df_merged['Revenue_Fact'] - \
        df_merged['Revenue_Plan']
    df_merged['Revenue_Diff_Pct'] = (
        safe_divide(
            df_merged['Revenue_Diff'],
            df_merged['Revenue_Plan']) *
        100).round(2)

    df_merged['Units_Diff'] = df_merged['Units_Fact'] - df_merged['Units_Plan']
    df_merged['Units_Diff_Pct'] = (
        safe_divide(
            df_merged['Units_Diff'],
            df_merged['Units_Plan']) *
        100).round(2)

    return df_merged, df_fact

# Функция для безопасного деления (защита от деления на ноль)


def safe_divide(numerator, denominator, default=0):
    """Безопасное деление с защитой от деления на ноль"""
    if isinstance(numerator, pd.Series) and isinstance(denominator, pd.Series):
        return numerator.div(denominator).replace(
            [np.inf, -np.inf], default).fillna(default)
    elif isinstance(numerator, (int, float)) and isinstance(denominator, (int, float)):
        return numerator / denominator if denominator != 0 else default
    else:
        # Для смешанных типов
        result = np.where(denominator != 0, numerator / denominator, default)
        return result

# Функция для расчета финансовых метрик


def calculate_financial_metrics(df_merged, df_fact_detailed):
    """Расчет расширенных финансовых метрик"""
    metrics = {}

    # Общая статистика
    metrics['total_revenue_plan'] = df_merged['Revenue_Plan'].sum()
    metrics['total_revenue_fact'] = df_merged['Revenue_Fact'].sum()
    metrics['total_units_plan'] = df_merged['Units_Plan'].sum()
    metrics['total_units_fact'] = df_merged['Units_Fact'].sum()

    # Отклонения
    metrics['revenue_variance'] = metrics['total_revenue_fact'] - \
        metrics['total_revenue_plan']
    metrics['revenue_variance_pct'] = safe_divide(
        metrics['revenue_variance'], metrics['total_revenue_plan'], 0) * 100
    metrics['units_variance'] = metrics['total_units_fact'] - \
        metrics['total_units_plan']
    metrics['units_variance_pct'] = safe_divide(
        metrics['units_variance'], metrics['total_units_plan'], 0) * 100

    # Средний чек
    metrics['avg_check_plan'] = safe_divide(
        metrics['total_revenue_plan'], metrics['total_units_plan'], 0)
    metrics['avg_check_fact'] = safe_divide(
        metrics['total_revenue_fact'], metrics['total_units_fact'], 0)
    metrics['avg_check_diff'] = metrics['avg_check_fact'] - \
        metrics['avg_check_plan']
    metrics['avg_check_diff_pct'] = safe_divide(
        metrics['avg_check_diff'], metrics['avg_check_plan'], 0) * 100

    # ROI и выполнение плана
    metrics['plan_achievement'] = safe_divide(
        metrics['total_revenue_fact'],
        metrics['total_revenue_plan'],
        0) * 100

    # Количество магазинов
    metrics['total_stores'] = df_merged['Magazin'].nunique()
    metrics['stores_above_plan'] = len(df_merged.groupby('Magazin').agg({
        'Revenue_Fact': 'sum',
        'Revenue_Plan': 'sum'
    }).query('Revenue_Fact > Revenue_Plan'))

    # Конверсия (процент магазинов выполнивших план)
    metrics['store_success_rate'] = safe_divide(
        metrics['stores_above_plan'], metrics['total_stores'], 0) * 100

    return metrics

# Функция для ABC анализа


def perform_abc_analysis(df_merged):
    """ABC анализ магазинов по выручке"""
    store_revenue = df_merged.groupby('Magazin').agg({
        'Revenue_Fact': 'sum',
        'Revenue_Plan': 'sum'
    }).reset_index()

    # Сортировка по факту выручки
    store_revenue = store_revenue.sort_values('Revenue_Fact', ascending=False)
    store_revenue['Revenue_Cumsum'] = store_revenue['Revenue_Fact'].cumsum()
    total_revenue = store_revenue['Revenue_Fact'].sum()
    store_revenue['Revenue_Cumsum_Pct'] = safe_divide(
        store_revenue['Revenue_Cumsum'], total_revenue, 0) * 100

    # Категории ABC
    def assign_abc_category(pct):
        if pct <= 80:
            return 'A'
        elif pct <= 95:
            return 'B'
        else:
            return 'C'

    store_revenue['ABC_Category'] = store_revenue['Revenue_Cumsum_Pct'].apply(
        assign_abc_category)
    store_revenue['Performance'] = safe_divide(
        store_revenue['Revenue_Fact'],
        store_revenue['Revenue_Plan'],
        0) * 100

    return store_revenue

# Функция для форматирования чисел


def format_number(num, decimals=0):
    """Форматирование чисел с разделителями"""
    if decimals == 0:
        return f"{int(num):,}".replace(',', ' ')
    else:
        return f"{num:,.{decimals}f}".replace(',', ' ')


# Функции для прогнозирования и планирования

def calculate_growth_rate(df_merged, df_fact_detailed):
    """Расчет темпа роста продаж"""
    # Анализ продаж по месяцам
    monthly_sales = df_fact_detailed.groupby('Month').agg({
        'Sum': 'sum',
        'Qty': 'sum'
    }).reset_index()
    monthly_sales = monthly_sales.sort_values('Month')

    if len(monthly_sales) < 2:
        return 0, monthly_sales

    # Средний темп роста
    growth_rates = []
    for i in range(1, len(monthly_sales)):
        prev_revenue = monthly_sales.iloc[i-1]['Sum']
        curr_revenue = monthly_sales.iloc[i]['Sum']
        if prev_revenue > 0:
            growth_rate = ((curr_revenue - prev_revenue) / prev_revenue) * 100
            growth_rates.append(growth_rate)

    avg_growth_rate = np.mean(growth_rates) if growth_rates else 0
    return avg_growth_rate, monthly_sales


# Продвинутые модели прогнозирования

def calculate_forecast_accuracy(y_true, y_pred):
    """Расчет метрик точности прогноза"""
    # Убираем нулевые значения для корректного расчета MAPE
    mask = y_true != 0
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]

    if len(y_true_filtered) == 0:
        return {'MAPE': 0, 'RMSE': 0, 'MAE': 0, 'RMSE_Pct': 0, 'MAE_Pct': 0, 'Mean_Value': 0}

    # Средняя выручка (для расчета процентов)
    mean_value = np.mean(y_true)

    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)) * 100

    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rmse_pct = (rmse / mean_value * 100) if mean_value > 0 else 0

    # MAE (Mean Absolute Error)
    mae = mean_absolute_error(y_true, y_pred)
    mae_pct = (mae / mean_value * 100) if mean_value > 0 else 0

    return {
        'MAPE': round(mape, 2),
        'RMSE': round(rmse, 2),
        'MAE': round(mae, 2),
        'RMSE_Pct': round(rmse_pct, 2),
        'MAE_Pct': round(mae_pct, 2),
        'Mean_Value': round(mean_value, 2)
    }


def forecast_linear_regression(monthly_sales, periods=3):
    """Прогнозирование с использованием линейной регрессии"""
    if len(monthly_sales) < 2:
        return None, None

    # Подготовка данных
    X = np.arange(len(monthly_sales)).reshape(-1, 1)
    y_revenue = monthly_sales['Sum'].values
    y_units = monthly_sales['Qty'].values

    # Обучение модели
    model_revenue = LinearRegression()
    model_revenue.fit(X, y_revenue)

    model_units = LinearRegression()
    model_units.fit(X, y_units)

    # Расчет точности на исторических данных
    y_pred_revenue = model_revenue.predict(X)
    accuracy = calculate_forecast_accuracy(y_revenue, y_pred_revenue)

    # Прогноз на будущее
    future_X = np.arange(len(monthly_sales), len(monthly_sales) + periods).reshape(-1, 1)
    forecast_revenue = model_revenue.predict(future_X)
    forecast_units = model_units.predict(future_X)

    # Защита от отрицательных значений
    forecast_revenue = np.maximum(forecast_revenue, 0)
    forecast_units = np.maximum(forecast_units, 0)

    return {
        'revenue': forecast_revenue,
        'units': forecast_units,
        'model_name': 'Линейная регрессия'
    }, accuracy


def forecast_polynomial_regression(monthly_sales, periods=3, degree=2):
    """Прогнозирование с использованием полиномиальной регрессии"""
    if len(monthly_sales) < degree + 1:
        return None, None

    # Подготовка данных
    X = np.arange(len(monthly_sales)).reshape(-1, 1)
    y_revenue = monthly_sales['Sum'].values
    y_units = monthly_sales['Qty'].values

    # Полиномиальные признаки
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)

    # Обучение модели
    model_revenue = LinearRegression()
    model_revenue.fit(X_poly, y_revenue)

    model_units = LinearRegression()
    model_units.fit(X_poly, y_units)

    # Расчет точности
    y_pred_revenue = model_revenue.predict(X_poly)
    accuracy = calculate_forecast_accuracy(y_revenue, y_pred_revenue)

    # Прогноз на будущее
    future_X = np.arange(len(monthly_sales), len(monthly_sales) + periods).reshape(-1, 1)
    future_X_poly = poly_features.transform(future_X)
    forecast_revenue = model_revenue.predict(future_X_poly)
    forecast_units = model_units.predict(future_X_poly)

    # Защита от отрицательных значений
    forecast_revenue = np.maximum(forecast_revenue, 0)
    forecast_units = np.maximum(forecast_units, 0)

    return {
        'revenue': forecast_revenue,
        'units': forecast_units,
        'model_name': f'Полиномиальная регрессия (степень {degree})'
    }, accuracy


def forecast_exponential_smoothing(monthly_sales, periods=3, alpha=0.3):
    """Прогнозирование с использованием экспоненциального сглаживания"""
    if len(monthly_sales) < 2:
        return None, None

    revenue_data = monthly_sales['Sum'].values
    units_data = monthly_sales['Qty'].values

    # Простое экспоненциальное сглаживание
    def exp_smoothing(data, alpha):
        result = [data[0]]
        for i in range(1, len(data)):
            result.append(alpha * data[i] + (1 - alpha) * result[i-1])
        return np.array(result)

    smoothed_revenue = exp_smoothing(revenue_data, alpha)

    # Расчет точности
    accuracy = calculate_forecast_accuracy(revenue_data, smoothed_revenue)

    # Прогноз - продолжаем тренд
    last_value_revenue = smoothed_revenue[-1]
    last_value_units = units_data[-1]

    # Рассчитываем средний тренд
    if len(smoothed_revenue) >= 2:
        trend_revenue = (smoothed_revenue[-1] - smoothed_revenue[-2])
        trend_units = (units_data[-1] - units_data[-2]) if len(units_data) >= 2 else 0
    else:
        trend_revenue = 0
        trend_units = 0

    forecast_revenue = []
    forecast_units = []

    for i in range(1, periods + 1):
        forecast_revenue.append(max(0, last_value_revenue + trend_revenue * i))
        forecast_units.append(max(0, last_value_units + trend_units * i))

    return {
        'revenue': np.array(forecast_revenue),
        'units': np.array(forecast_units),
        'model_name': f'Экспоненциальное сглаживание (α={alpha})'
    }, accuracy


def forecast_weighted_moving_average(monthly_sales, periods=3, window=3):
    """Прогнозирование с использованием взвешенного скользящего среднего"""
    if len(monthly_sales) < window:
        return None, None

    revenue_data = monthly_sales['Sum'].values
    units_data = monthly_sales['Qty'].values

    # Веса (больший вес для более свежих данных)
    weights = np.arange(1, window + 1)
    weights = weights / weights.sum()

    # Расчет WMA для исторических данных
    wma_revenue = []
    for i in range(len(revenue_data)):
        if i < window - 1:
            wma_revenue.append(revenue_data[i])
        else:
            wma_revenue.append(np.sum(weights * revenue_data[i-window+1:i+1]))

    wma_revenue = np.array(wma_revenue)

    # Расчет точности
    accuracy = calculate_forecast_accuracy(revenue_data, wma_revenue)

    # Прогноз
    forecast_revenue = []
    forecast_units = []

    last_values_revenue = revenue_data[-window:]
    last_values_units = units_data[-window:]

    for i in range(periods):
        next_revenue = np.sum(weights * last_values_revenue)
        next_units = np.sum(weights * last_values_units)

        forecast_revenue.append(max(0, next_revenue))
        forecast_units.append(max(0, next_units))

        # Обновляем окно для следующей итерации
        last_values_revenue = np.append(last_values_revenue[1:], next_revenue)
        last_values_units = np.append(last_values_units[1:], next_units)

    return {
        'revenue': np.array(forecast_revenue),
        'units': np.array(forecast_units),
        'model_name': f'Взвешенное скользящее среднее (окно={window})'
    }, accuracy


def forecast_ensemble(monthly_sales, periods=3):
    """Ансамблевое прогнозирование - среднее всех моделей"""
    forecasts = []
    accuracies = []

    # Собираем прогнозы всех моделей
    models = [
        forecast_linear_regression(monthly_sales, periods),
        forecast_polynomial_regression(monthly_sales, periods, degree=2),
        forecast_exponential_smoothing(monthly_sales, periods, alpha=0.3),
        forecast_weighted_moving_average(monthly_sales, periods, window=min(3, len(monthly_sales)))
    ]

    valid_forecasts_revenue = []
    valid_forecasts_units = []

    for model_result, accuracy in models:
        if model_result is not None:
            valid_forecasts_revenue.append(model_result['revenue'])
            valid_forecasts_units.append(model_result['units'])
            accuracies.append(accuracy)

    if not valid_forecasts_revenue:
        return None, None

    # Среднее всех прогнозов
    ensemble_revenue = np.mean(valid_forecasts_revenue, axis=0)
    ensemble_units = np.mean(valid_forecasts_units, axis=0)

    # Средняя точность
    avg_accuracy = {
        'MAPE': round(np.mean([acc['MAPE'] for acc in accuracies]), 2),
        'RMSE': round(np.mean([acc['RMSE'] for acc in accuracies]), 2),
        'MAE': round(np.mean([acc['MAE'] for acc in accuracies]), 2),
        'RMSE_Pct': round(np.mean([acc['RMSE_Pct'] for acc in accuracies]), 2),
        'MAE_Pct': round(np.mean([acc['MAE_Pct'] for acc in accuracies]), 2),
        'Mean_Value': round(np.mean([acc['Mean_Value'] for acc in accuracies]), 2)
    }

    return {
        'revenue': ensemble_revenue,
        'units': ensemble_units,
        'model_name': 'Ансамбль моделей'
    }, avg_accuracy


def forecast_with_multiple_models(df_merged, df_fact_detailed, periods=3):
    """Прогнозирование с использованием нескольких моделей и выбором лучшей"""
    avg_growth_rate, monthly_sales = calculate_growth_rate(df_merged, df_fact_detailed)

    if monthly_sales.empty or len(monthly_sales) < 2:
        return None

    last_month_date = pd.Period(monthly_sales.iloc[-1]['Month'])

    # Запускаем все модели
    all_models = {
        'linear': forecast_linear_regression(monthly_sales, periods),
        'polynomial': forecast_polynomial_regression(monthly_sales, periods, degree=2),
        'exponential': forecast_exponential_smoothing(monthly_sales, periods, alpha=0.3),
        'wma': forecast_weighted_moving_average(monthly_sales, periods, window=min(3, len(monthly_sales))),
        'ensemble': forecast_ensemble(monthly_sales, periods)
    }

    # Собираем результаты
    results = []
    for model_key, (model_result, accuracy) in all_models.items():
        if model_result is not None and accuracy is not None:
            model_forecasts = []
            for i in range(periods):
                forecast_month = (last_month_date + i + 1).strftime('%Y-%m')
                model_forecasts.append({
                    'Month': forecast_month,
                    'Forecast_Revenue': model_result['revenue'][i],
                    'Forecast_Units': int(model_result['units'][i]),
                    'Model': model_result['model_name'],
                    'Model_Key': model_key,
                    'MAPE': accuracy['MAPE'],
                    'RMSE': accuracy['RMSE'],
                    'MAE': accuracy['MAE'],
                    'RMSE_Pct': accuracy['RMSE_Pct'],
                    'MAE_Pct': accuracy['MAE_Pct'],
                    'Mean_Value': accuracy['Mean_Value']
                })
            results.extend(model_forecasts)

    if not results:
        return None

    return pd.DataFrame(results)


def apply_scenario(forecast_df, scenario='realistic'):
    """Применение сценария к прогнозу"""
    if forecast_df is None or forecast_df.empty:
        return None

    # Коэффициенты для сценариев
    scenario_factors = {
        'optimistic': 1.20,      # +20%
        'realistic': 1.00,       # без изменений
        'pessimistic': 0.85      # -15%
    }

    scenario_names = {
        'optimistic': 'Оптимистичный',
        'realistic': 'Реальный',
        'pessimistic': 'Пессимистичный'
    }

    factor = scenario_factors.get(scenario, 1.0)
    df_scenario = forecast_df.copy()

    df_scenario['Forecast_Revenue'] = df_scenario['Forecast_Revenue'] * factor
    df_scenario['Forecast_Units'] = (df_scenario['Forecast_Units'] * factor).astype(int)
    df_scenario['Scenario'] = scenario_names[scenario]
    df_scenario['Scenario_Factor'] = factor

    return df_scenario


def forecast_next_period(df_merged, df_fact_detailed, periods=3):
    """Прогнозирование продаж на следующие периоды"""
    avg_growth_rate, monthly_sales = calculate_growth_rate(df_merged, df_fact_detailed)

    if monthly_sales.empty:
        return None

    last_month_sales = monthly_sales.iloc[-1]['Sum']
    last_month_units = monthly_sales.iloc[-1]['Qty']
    last_month_date = pd.Period(monthly_sales.iloc[-1]['Month'])

    forecasts = []
    for i in range(1, periods + 1):
        forecast_month = (last_month_date + i).strftime('%Y-%m')
        # Прогноз с учетом роста
        forecast_revenue = last_month_sales * ((1 + avg_growth_rate / 100) ** i)
        forecast_units = last_month_units * ((1 + avg_growth_rate / 100) ** i)

        forecasts.append({
            'Month': forecast_month,
            'Forecast_Revenue': forecast_revenue,
            'Forecast_Units': int(forecast_units),
            'Growth_Rate': avg_growth_rate
        })

    return pd.DataFrame(forecasts)


def generate_plan_recommendations(df_merged, df_fact_detailed, financial_metrics, abc_analysis):
    """Генерация рекомендаций для планирования"""
    recommendations = []

    # 1. Анализ выполнения плана
    if financial_metrics['plan_achievement'] < 90:
        recommendations.append({
            'priority': 'Высокий',
            'category': 'Выполнение плана',
            'issue': f"Общее выполнение плана составляет {financial_metrics['plan_achievement']:.1f}%",
            'recommendation': 'Снизить плановые показатели на 10-15% или усилить маркетинговую поддержку',
            'impact': 'Высокий'
        })
    elif financial_metrics['plan_achievement'] > 110:
        recommendations.append({
            'priority': 'Средний',
            'category': 'Выполнение плана',
            'issue': f"План перевыполнен на {financial_metrics['plan_achievement'] - 100:.1f}%",
            'recommendation': 'Пересмотреть плановые показатели в сторону увеличения на 5-10%',
            'impact': 'Средний'
        })

    # 2. Анализ по сегментам
    segment_performance = df_merged.groupby('Segment').agg({
        'Revenue_Fact': 'sum',
        'Revenue_Plan': 'sum'
    }).reset_index()
    segment_performance['Achievement'] = safe_divide(
        segment_performance['Revenue_Fact'],
        segment_performance['Revenue_Plan']
    ) * 100

    underperforming_segments = segment_performance[segment_performance['Achievement'] < 85]
    for _, seg in underperforming_segments.iterrows():
        recommendations.append({
            'priority': 'Высокий',
            'category': 'Сегментация',
            'issue': f"Сегмент '{seg['Segment']}' показывает низкое выполнение: {seg['Achievement']:.1f}%",
            'recommendation': f"Провести анализ ассортимента в сегменте {seg['Segment']}, рассмотреть промо-акции",
            'impact': 'Высокий'
        })

    # 3. Анализ ABC категорий
    category_c = abc_analysis[abc_analysis['ABC_Category'] == 'C']
    if len(category_c) > 0:
        total_c_revenue = category_c['Revenue_Fact'].sum()
        recommendations.append({
            'priority': 'Средний',
            'category': 'ABC анализ',
            'issue': f"Категория C содержит {len(category_c)} магазинов с низкой эффективностью",
            'recommendation': f"Рассмотреть оптимизацию работы или закрытие неэффективных точек",
            'impact': 'Средний'
        })

    # 4. Анализ среднего чека
    if financial_metrics['avg_check_diff_pct'] < -10:
        recommendations.append({
            'priority': 'Высокий',
            'category': 'Средний чек',
            'issue': f"Средний чек снизился на {abs(financial_metrics['avg_check_diff_pct']):.1f}%",
            'recommendation': 'Внедрить up-selling и cross-selling стратегии, обучить персонал',
            'impact': 'Высокий'
        })

    # 5. Успешность магазинов
    if financial_metrics['store_success_rate'] < 50:
        recommendations.append({
            'priority': 'Критический',
            'category': 'Эффективность сети',
            'issue': f"Только {financial_metrics['store_success_rate']:.0f}% магазинов выполняют план",
            'recommendation': 'Провести аудит неэффективных точек, пересмотреть систему мотивации',
            'impact': 'Критический'
        })

    return pd.DataFrame(recommendations) if recommendations else None


def create_smart_plan(df_merged, df_fact_detailed, forecast_periods=3, adjustment_factor=1.0):
    """
    Создание умного плана на основе исторических данных
    adjustment_factor: коэффициент корректировки (1.0 = без изменений, 1.1 = +10%, 0.9 = -10%)
    """
    # Получаем прогноз
    forecast_df = forecast_next_period(df_merged, df_fact_detailed, forecast_periods)

    if forecast_df is None or forecast_df.empty:
        return None

    # Анализ по магазинам и сегментам
    store_segment_avg = df_merged.groupby(['Magazin', 'Segment']).agg({
        'Revenue_Fact': 'mean',
        'Units_Fact': 'mean'
    }).reset_index()

    # Генерация плана для каждого магазина/сегмента
    smart_plan = []
    for _, row in store_segment_avg.iterrows():
        for _, forecast in forecast_df.iterrows():
            # Доля магазина/сегмента от общих продаж
            total_revenue = store_segment_avg['Revenue_Fact'].sum()
            store_segment_share = safe_divide(row['Revenue_Fact'], total_revenue, 0)

            # Прогнозный план с учетом доли и коэффициента корректировки
            planned_revenue = forecast['Forecast_Revenue'] * store_segment_share * adjustment_factor
            planned_units = int(forecast['Forecast_Units'] * store_segment_share * adjustment_factor)

            smart_plan.append({
                'Magazin': row['Magazin'],
                'Segment': row['Segment'],
                'Month': forecast['Month'],
                'Revenue_Plan': round(planned_revenue, 2),
                'Units_Plan': planned_units,
                'Based_on': 'Исторические данные + прогноз',
                'Growth_Rate': forecast['Growth_Rate']
            })

    return pd.DataFrame(smart_plan)

# Главная функция


def main():

    # Заголовок
    st.title("👓 План/Факт Продаж Оптика")

    # Sidebar - фильтры
    st.sidebar.header("⚙️ Фильтры")

    # Загрузка данных
    data_source = st.sidebar.radio(
        "📂 Источник данных",
        options=["Демо-данные", "Excel/CSV файлы", "Google Sheets"],
        index=0
    )

    if data_source == "Демо-данные":
        df_fact, df_plan = generate_demo_data()

    elif data_source == "Excel/CSV файлы":
        st.sidebar.subheader("📊 Загрузка файлов")

        st.sidebar.info(
            "📋 **Требуемые колонки для файла ФАКТ:**\n"
            "- Magazin (название магазина)\n"
            "- Datasales (дата продажи)\n"
            "- Segment (сегмент товара)\n"
            "- Price (цена)\n"
            "- Qty (количество)\n"
            "- Sum (сумма)\n\n"
            "📋 **Требуемые колонки для файла ПЛАН:**\n"
            "- Magazin (название магазина)\n"
            "- Segment (сегмент товара)\n"
            "- Month (месяц в формате YYYY-MM)\n"
            "- Revenue_Plan (план выручки)\n"
            "- Units_Plan (план штук)"
        )

        st.sidebar.markdown("---")

        fact_file = st.sidebar.file_uploader(
            "📁 Загрузить файл ФАКТ",
            type=['xlsx', 'xls', 'csv'],
            help="Форматы: Excel (.xlsx, .xls) или CSV"
        )

        plan_file = st.sidebar.file_uploader(
            "📁 Загрузить файл ПЛАН",
            type=['xlsx', 'xls', 'csv'],
            help="Форматы: Excel (.xlsx, .xls) или CSV"
        )

        if fact_file and plan_file:
            with st.spinner("Загрузка файлов..."):
                df_fact, df_plan = load_data_from_excel(fact_file, plan_file)
        else:
            st.info("👈 Загрузите оба файла (ФАКТ и ПЛАН) для начала анализа")
            return

    else:  # Google Sheets
        st.sidebar.subheader("Google Sheets")

        st.sidebar.info(
            "📌 Как получить ссылку:\n1. Открой нужный лист в Google Sheets\n2. Скопируй URL из адресной строки")

        # Ссылка на лист План
        plan_url = st.sidebar.text_input(
            "🔗 Ссылка на лист План",
            value="https://docs.google.com/spreadsheets/d/1lJLON5N_EKQ5ICv0Pprp5DamP1tNAhBIph4uEoWC04Q/edit#gid=103045414",
            placeholder="https://docs.google.com/.../edit#gid=...",
            help="Открой лист 'Plan' в Google Sheets и скопируй URL из браузера"
        )

        # Ссылка на лист Факт
        fact_url = st.sidebar.text_input(
            "🔗 Ссылка на лист Факт",
            value="https://docs.google.com/spreadsheets/d/1lJLON5N_EKQ5ICv0Pprp5DamP1tNAhBIph4uEoWC04Q/edit#gid=1144131206",
            placeholder="https://docs.google.com/.../edit#gid=...",
            help="Открой лист 'Fact' в Google Sheets и скопируй URL из браузера"
        )

        if st.sidebar.button("🔄 Загрузить данные", use_container_width=True):
            with st.spinner("Загрузка из Google Sheets..."):
                df_fact, df_plan = load_data_from_sheets(plan_url, fact_url)
        else:
            st.info(
                "👈 Вставь ссылки на листы Plan и Fact, затем нажми 'Загрузить данные'")
            return

    if df_fact is None or df_plan is None:
        st.warning(
            "Данные не загружены. Используйте демо-данные или проверьте подключение.")
        return

    # Подготовка данных
    df_merged, df_fact_detailed = prepare_data(df_fact, df_plan)

    # Проверка результата подготовки данных
    if df_merged is None or df_fact_detailed is None:
        st.error("❌ Ошибка подготовки данных. Проверьте структуру загруженных файлов.")
        return

    if df_merged.empty:
        st.warning("⚠️ После обработки данные оказались пустыми. Проверьте соответствие периодов в Плане и Факте.")
        return

    # Фильтр по месяцам
    available_months = sorted(df_merged['Month'].unique())
    selected_months = st.sidebar.multiselect(
        "Выберите месяцы",
        options=available_months,
        default=available_months
    )

    # Фильтр по сегментам
    available_segments = sorted(df_merged['Segment'].unique())
    selected_segments = st.sidebar.multiselect(
        "Выберите сегменты",
        options=available_segments,
        default=available_segments
    )

    # Применение фильтров
    df_filtered = df_merged[
        (df_merged['Month'].isin(selected_months)) &
        (df_merged['Segment'].isin(selected_segments))
    ]

    # Проверка фильтрованных данных
    if df_filtered.empty:
        st.warning("⚠️ По выбранным фильтрам нет данных. Измените параметры фильтрации.")
        return

    # Alerts
    st.sidebar.markdown("---")
    st.sidebar.header("🚨 Alerts")

    alerts = df_filtered[abs(df_filtered['Revenue_Diff_Pct']) > 10].copy()
    alerts = alerts.sort_values('Revenue_Diff_Pct')

    if len(alerts) > 0:
        st.sidebar.error(f"Критических отклонений: {len(alerts)}")
        for idx, row in alerts.head(10).iterrows():
            emoji = "🔴" if row['Revenue_Diff_Pct'] < 0 else "🟢"
            st.sidebar.write(
                f"{emoji} **{row['Magazin']}** ({row['Segment']}): {row['Revenue_Diff_Pct']:+.1f}%"
            )
    else:
        st.sidebar.success("Критических отклонений нет")

    # Расчет финансовых метрик
    financial_metrics = calculate_financial_metrics(
        df_filtered, df_fact_detailed)
    abc_analysis = perform_abc_analysis(df_filtered)

    # Основной контент - Tabs
    tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📋 Executive Summary",
        "📊 Сводка",
        "🏪 По магазинам",
        "📦 По сегментам",
        "📈 Динамика",
        "🎯 ABC Анализ",
        "📊 Общая сводка анализа",
        "🎯 Планирование"
    ])

    # TAB 0: Executive Summary
    with tab0:
        st.header("📋 Управленческий отчет: Executive Summary")
        st.markdown("---")

        # Период отчета
        period_range = f"{min(selected_months)} - {max(selected_months)}"
        st.subheader(f"Период: {period_range}")

        # Ключевые финансовые метрики
        st.markdown("### 💰 Ключевые финансовые показатели")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "План выручки",
                f"{format_number(financial_metrics['total_revenue_plan'])} ₴",
                help="Плановая выручка за период"
            )

        with col2:
            st.metric(
                "Факт выручки",
                f"{format_number(financial_metrics['total_revenue_fact'])} ₴",
                delta=f"{financial_metrics['revenue_variance_pct']:+.1f}%",
                delta_color="normal",
                help="Фактическая выручка за период"
            )

        with col3:
            st.metric(
                "Выполнение плана",
                f"{financial_metrics['plan_achievement']:.1f}%",
                delta=f"{financial_metrics['plan_achievement'] - 100:+.1f}%",
                delta_color="normal",
                help="Процент выполнения плана по выручке"
            )

        with col4:
            st.metric(
                "Средний чек",
                f"{format_number(financial_metrics['avg_check_fact'])} ₴",
                delta=f"{financial_metrics['avg_check_diff_pct']:+.1f}%",
                delta_color="normal",
                help="Средняя стоимость продажи"
            )

        with col5:
            st.metric(
                "Успешность точек",
                f"{financial_metrics['store_success_rate']:.0f}%",
                delta=f"{financial_metrics['stores_above_plan']} из {financial_metrics['total_stores']}",
                help="Процент магазинов, выполнивших план"
            )

        st.markdown("---")

        # Финансовый анализ
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### 📊 Анализ отклонений")

            # Waterfall chart для выручки
            fig_waterfall = go.Figure(go.Waterfall(
                name="Выручка",
                orientation="v",
                measure=["absolute", "relative", "total"],
                x=["План", "Отклонение", "Факт"],
                y=[
                    financial_metrics['total_revenue_plan'],
                    financial_metrics['revenue_variance'],
                    financial_metrics['total_revenue_fact']
                ],
                text=[
                    f"{format_number(financial_metrics['total_revenue_plan'])} ₴",
                    f"{financial_metrics['revenue_variance_pct']:+.1f}%",
                    f"{format_number(financial_metrics['total_revenue_fact'])} ₴"
                ],
                textposition="outside",
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                decreasing={"marker": {"color": "#ff6b6b"}},
                increasing={"marker": {"color": "#51cf66"}},
                totals={"marker": {"color": "#4dabf7"}}
            ))

            fig_waterfall.update_layout(
                title="План vs Факт: Waterfall анализ",
                height=400,
                showlegend=False,
                yaxis_title="Выручка (₴)"
            )

            st.plotly_chart(fig_waterfall, use_container_width=True)

        with col2:
            st.markdown("### 🎯 Ключевые выводы")

            # Автоматические выводы на основе данных
            if financial_metrics['revenue_variance_pct'] > 5:
                st.success(
                    f"✅ План перевыполнен на {financial_metrics['revenue_variance_pct']:.1f}% (+{format_number(financial_metrics['revenue_variance'])} ₴)")
            elif financial_metrics['revenue_variance_pct'] >= -5:
                st.info(
                    f"ℹ️ План выполнен на {financial_metrics['plan_achievement']:.1f}%")
            else:
                st.warning(
                    f"⚠️ Недовыполнение плана на {abs(financial_metrics['revenue_variance_pct']):.1f}% ({format_number(financial_metrics['revenue_variance'])} ₴)")

            st.markdown(f"""
            **Операционные показатели:**
            - Продано единиц (план): **{format_number(financial_metrics['total_units_plan'])}** шт
            - Продано единиц (факт): **{format_number(financial_metrics['total_units_fact'])}** шт ({financial_metrics['units_variance_pct']:+.1f}%)
            - Средний чек (план): **{format_number(financial_metrics['avg_check_plan'])}** ₴
            - Средний чек (факт): **{format_number(financial_metrics['avg_check_fact'])}** ₴ ({financial_metrics['avg_check_diff_pct']:+.1f}%)

            **Эффективность сети:**
            - Всего торговых точек: **{financial_metrics['total_stores']}**
            - Выполнили план: **{financial_metrics['stores_above_plan']}** ({financial_metrics['store_success_rate']:.0f}%)
            - Не выполнили план: **{financial_metrics['total_stores'] - financial_metrics['stores_above_plan']}**
            """)

            # Рекомендации
            st.markdown("**💡 Рекомендации:**")

            if financial_metrics['revenue_variance_pct'] < -5:
                st.markdown("- 🔴 Провести анализ неэффективных точек")
                st.markdown("- 🔴 Пересмотреть ассортиментную матрицу")
                st.markdown("- 🔴 Усилить маркетинговую активность")
            elif financial_metrics['store_success_rate'] < 60:
                st.markdown("- 🟡 Провести обучение персонала отстающих точек")
                st.markdown("- 🟡 Оптимизировать систему мотивации")
            else:
                st.markdown("- 🟢 Масштабировать успешные практики")
                st.markdown("- 🟢 Поддерживать текущий уровень эффективности")

        st.markdown("---")

        # Сегментный анализ
        st.markdown("### 📦 Анализ по сегментам")

        segment_performance = df_filtered.groupby('Segment').agg({
            'Revenue_Plan': 'sum',
            'Revenue_Fact': 'sum',
            'Units_Plan': 'sum',
            'Units_Fact': 'sum'
        }).reset_index()

        segment_performance['Achievement_%'] = safe_divide(
            segment_performance['Revenue_Fact'],
            segment_performance['Revenue_Plan']
        ) * 100

        segment_performance['Avg_Check'] = safe_divide(
            segment_performance['Revenue_Fact'],
            segment_performance['Units_Fact']
        )

        # Форматирование таблицы
        segment_table = segment_performance.copy()
        segment_table['Revenue_Plan'] = segment_table['Revenue_Plan'].apply(
            lambda x: f"{format_number(x)} ₴")
        segment_table['Revenue_Fact'] = segment_table['Revenue_Fact'].apply(
            lambda x: f"{format_number(x)} ₴")
        segment_table['Units_Plan'] = segment_table['Units_Plan'].apply(
            lambda x: f"{format_number(x)} шт")
        segment_table['Units_Fact'] = segment_table['Units_Fact'].apply(
            lambda x: f"{format_number(x)} шт")
        segment_table['Achievement_%'] = segment_table['Achievement_%'].apply(
            lambda x: f"{x:.1f}%")
        segment_table['Avg_Check'] = segment_table['Avg_Check'].apply(
            lambda x: f"{format_number(x)} ₴")

        st.dataframe(segment_table, use_container_width=True, height=200)

    # TAB 1: Сводка
    with tab1:
        # KPI карточки
        col1, col2, col3, col4 = st.columns(4)

        total_revenue_plan = df_filtered['Revenue_Plan'].sum()
        total_revenue_fact = df_filtered['Revenue_Fact'].sum()
        total_revenue_diff = total_revenue_fact - total_revenue_plan
        total_revenue_diff_pct = (
            total_revenue_diff /
            total_revenue_plan *
            100) if total_revenue_plan > 0 else 0

        total_units_plan = df_filtered['Units_Plan'].sum()
        total_units_fact = df_filtered['Units_Fact'].sum()
        total_units_diff = total_units_fact - total_units_plan
        total_units_diff_pct = (
            total_units_diff /
            total_units_plan *
            100) if total_units_plan > 0 else 0

        with col1:
            st.metric(
                "План Выручка",
                f"{format_number(total_revenue_plan)} ₴",
                delta=None
            )

        with col2:
            st.metric(
                "Факт Выручка",
                f"{format_number(total_revenue_fact)} ₴",
                delta=f"{total_revenue_diff_pct:+.1f}%",
                delta_color="normal"
            )

        with col3:
            st.metric(
                "План Штуки",
                f"{format_number(total_units_plan)} шт",
                delta=None
            )

        with col4:
            st.metric(
                "Факт Штуки",
                f"{format_number(total_units_fact)} шт",
                delta=f"{total_units_diff_pct:+.1f}%",
                delta_color="normal"
            )

        st.markdown("---")

        # График план vs факт по месяцам
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Выручка: План vs Факт")

            revenue_by_month = df_filtered.groupby('Month').agg({
                'Revenue_Plan': 'sum',
                'Revenue_Fact': 'sum'
            }).reset_index()

            fig_revenue = go.Figure()
            fig_revenue.add_trace(go.Bar(
                x=revenue_by_month['Month'],
                y=revenue_by_month['Revenue_Plan'],
                name='План',
                marker_color='lightblue'
            ))
            fig_revenue.add_trace(go.Bar(
                x=revenue_by_month['Month'],
                y=revenue_by_month['Revenue_Fact'],
                name='Факт',
                marker_color='darkblue'
            ))
            fig_revenue.update_layout(
                barmode='group',
                height=400,
                xaxis_title="Месяц",
                yaxis_title="Выручка (₴)",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1)
            )
            st.plotly_chart(fig_revenue, use_container_width=True)

        with col2:
            st.subheader("Выполнение плана по сегментам")

            perf_by_segment = df_filtered.groupby('Segment').agg({
                'Revenue_Plan': 'sum',
                'Revenue_Fact': 'sum'
            }).reset_index()
            perf_by_segment['Performance'] = (
                safe_divide(
                    perf_by_segment['Revenue_Fact'],
                    perf_by_segment['Revenue_Plan']) * 100
            ).round(1)

            fig_segment = px.bar(
                perf_by_segment,
                x='Segment',
                y='Performance',
                text='Performance',
                color='Performance',
                color_continuous_scale=['red', 'yellow', 'green'],
                range_color=[80, 120]
            )
            fig_segment.update_traces(
                texttemplate='%{text:.1f}%',
                textposition='outside')
            fig_segment.update_layout(
                height=400,
                xaxis_title="Сегмент",
                yaxis_title="Выполнение плана (%)",
                showlegend=False
            )
            fig_segment.add_hline(
                y=100,
                line_dash="dash",
                line_color="gray",
                annotation_text="100%")
            st.plotly_chart(fig_segment, use_container_width=True)

    # TAB 2: По магазинам
    with tab2:
        st.subheader("Анализ по магазинам")

        # Группировка по магазинам
        store_summary = df_filtered.groupby('Magazin').agg({
            'Revenue_Plan': 'sum',
            'Revenue_Fact': 'sum',
            'Units_Plan': 'sum',
            'Units_Fact': 'sum'
        }).reset_index()

        store_summary['Revenue_Diff'] = store_summary['Revenue_Fact'] - \
            store_summary['Revenue_Plan']
        store_summary['Revenue_Diff_Pct'] = (
            safe_divide(
                store_summary['Revenue_Diff'],
                store_summary['Revenue_Plan']) * 100
        ).round(2)

        store_summary['Units_Diff'] = store_summary['Units_Fact'] - \
            store_summary['Units_Plan']
        store_summary['Units_Diff_Pct'] = (
            safe_divide(
                store_summary['Units_Diff'],
                store_summary['Units_Plan']) * 100
        ).round(2)

        # Форматирование таблицы
        def color_diff(val):
            if pd.isna(val):
                return ''
            color = 'green' if val >= 0 else 'red'
            return f'color: {color}'

        styled_table = store_summary.style.map(
            color_diff,
            subset=['Revenue_Diff_Pct', 'Units_Diff_Pct']
        ).format({
            'Revenue_Plan': lambda x: f"{format_number(x)} ₴",
            'Revenue_Fact': lambda x: f"{format_number(x)} ₴",
            'Revenue_Diff': lambda x: f"{format_number(x)} ₴",
            'Revenue_Diff_Pct': lambda x: f"{x:+.1f}%",
            'Units_Plan': lambda x: f"{format_number(x)} шт",
            'Units_Fact': lambda x: f"{format_number(x)} шт",
            'Units_Diff': lambda x: f"{format_number(x)} шт",
            'Units_Diff_Pct': lambda x: f"{x:+.1f}%"
        })

        st.dataframe(styled_table, use_container_width=True, height=400)

        # Детализация по выбранному магазину
        st.markdown("---")
        selected_store = st.selectbox(
            "Выберите магазин для детализации",
            options=sorted(df_filtered['Magazin'].unique())
        )

        if selected_store:
            store_detail = df_filtered[df_filtered['Magazin']
                                       == selected_store]

            col1, col2 = st.columns(2)

            with col1:
                st.subheader(f"Выполнение по сегментам: {selected_store}")

                fig_store_segment = px.bar(
                    store_detail,
                    x='Segment',
                    y=['Revenue_Plan', 'Revenue_Fact'],
                    barmode='group',
                    labels={'value': 'Выручка (₴)', 'variable': 'Тип'},
                    color_discrete_map={
                        'Revenue_Plan': 'lightblue',
                        'Revenue_Fact': 'darkblue'}
                )
                fig_store_segment.update_layout(height=350)
                st.plotly_chart(fig_store_segment, use_container_width=True)

            with col2:
                st.subheader("Детали по сегментам")
                st.dataframe(
                    store_detail[['Segment', 'Month', 'Revenue_Plan',
                                  'Revenue_Fact', 'Revenue_Diff_Pct']],
                    use_container_width=True,
                    height=350
                )

    # TAB 3: По сегментам
    with tab3:
        st.subheader("Анализ по сегментам")

        selected_segment = st.selectbox(
            "Выберите сегмент",
            options=sorted(df_filtered['Segment'].unique())
        )

        if selected_segment:
            segment_data = df_filtered[df_filtered['Segment']
                                       == selected_segment]

            segment_by_store = segment_data.groupby('Magazin').agg({
                'Revenue_Plan': 'sum',
                'Revenue_Fact': 'sum',
                'Units_Plan': 'sum',
                'Units_Fact': 'sum'
            }).reset_index()

            segment_by_store['Revenue_Diff_Pct'] = (
                safe_divide(
                    segment_by_store['Revenue_Fact'] -
                    segment_by_store['Revenue_Plan'],
                    segment_by_store['Revenue_Plan']
                ) * 100
            ).round(2)

            # График топ/худшие магазины
            segment_by_store_sorted = segment_by_store.sort_values(
                'Revenue_Diff_Pct')

            col1, col2 = st.columns(2)

            with col1:
                st.subheader(f"Топ-5 магазинов ({selected_segment})")
                top5 = segment_by_store_sorted.tail(5)
                fig_top = px.bar(
                    top5,
                    x='Revenue_Diff_Pct',
                    y='Magazin',
                    orientation='h',
                    text='Revenue_Diff_Pct',
                    color='Revenue_Diff_Pct',
                    color_continuous_scale='Greens'
                )
                fig_top.update_traces(
                    texttemplate='%{text:.1f}%',
                    textposition='outside')
                fig_top.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig_top, use_container_width=True)

            with col2:
                st.subheader(f"Худшие 5 магазинов ({selected_segment})")
                bottom5 = segment_by_store_sorted.head(5)
                fig_bottom = px.bar(
                    bottom5,
                    x='Revenue_Diff_Pct',
                    y='Magazin',
                    orientation='h',
                    text='Revenue_Diff_Pct',
                    color='Revenue_Diff_Pct',
                    color_continuous_scale='Reds'
                )
                fig_bottom.update_traces(
                    texttemplate='%{text:.1f}%',
                    textposition='outside')
                fig_bottom.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig_bottom, use_container_width=True)

    # TAB 4: Динамика
    with tab4:
        st.subheader("Динамика продаж")

        # Фильтр детализации
        time_grain = st.radio(
            "Детализация",
            options=['День', 'Неделя', 'Месяц'],
            horizontal=True
        )

        # Подготовка данных для графика
        if time_grain == 'День':
            df_fact_detailed['Period'] = df_fact_detailed['Datasales'].dt.strftime(
                '%Y-%m-%d')
        elif time_grain == 'Неделя':
            df_fact_detailed['Period'] = df_fact_detailed['Week']
        else:
            df_fact_detailed['Period'] = df_fact_detailed['Month']

        # Фильтр по выбранным месяцам
        df_fact_filtered = df_fact_detailed[df_fact_detailed['Month'].isin(
            selected_months)]

        daily_revenue = df_fact_filtered.groupby(
            'Period')['Sum'].sum().reset_index()
        daily_revenue.columns = ['Period', 'Revenue']

        fig_timeline = px.line(
            daily_revenue,
            x='Period',
            y='Revenue',
            markers=True,
            title=f"Динамика выручки ({time_grain.lower()})"
        )
        fig_timeline.update_layout(
            height=400,
            xaxis_title="Период",
            yaxis_title="Выручка (₴)"
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

        # Динамика по сегментам
        st.markdown("---")
        st.subheader("Динамика по сегментам")

        segment_timeline = df_fact_filtered.groupby(['Period', 'Segment'])[
            'Sum'].sum().reset_index()

        fig_segment_timeline = px.line(
            segment_timeline,
            x='Period',
            y='Sum',
            color='Segment',
            markers=True
        )
        fig_segment_timeline.update_layout(
            height=400,
            xaxis_title="Период",
            yaxis_title="Выручка (₴)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1)
        )
        st.plotly_chart(fig_segment_timeline, use_container_width=True)

    # TAB 5: ABC Анализ
    with tab5:
        st.header("🎯 ABC Анализ торговых точек")
        st.markdown("---")

        st.info("📊 **ABC анализ** - метод классификации торговых точек по вкладу в общую выручку:\n"
                "- **Категория A**: 80% выручки (ключевые точки)\n"
                "- **Категория B**: следующие 15% (важные точки)\n"
                "- **Категория C**: последние 5% (низкоэффективные точки)")

        # Статистика по категориям
        col1, col2, col3 = st.columns(3)

        abc_summary = abc_analysis.groupby('ABC_Category').agg({
            'Magazin': 'count',
            'Revenue_Fact': 'sum'
        }).reset_index()
        abc_summary.columns = ['Category', 'Store_Count', 'Revenue']
        total_stores_abc = abc_summary['Store_Count'].sum()
        total_revenue_abc = abc_summary['Revenue'].sum()

        for idx, category in enumerate(['A', 'B', 'C']):
            cat_data = abc_summary[abc_summary['Category'] == category]
            if len(cat_data) > 0:
                stores = cat_data['Store_Count'].values[0]
                revenue = cat_data['Revenue'].values[0]
                revenue_pct = safe_divide(revenue, total_revenue_abc, 0) * 100
                stores_pct = safe_divide(stores, total_stores_abc, 0) * 100

                if idx == 0:
                    with col1:
                        st.metric(
                            f"Категория {category}",
                            f"{stores} точек ({stores_pct:.0f}%)",
                            delta=f"{revenue_pct:.1f}% выручки",
                            help=f"Выручка: {format_number(revenue)} ₴"
                        )
                elif idx == 1:
                    with col2:
                        st.metric(
                            f"Категория {category}",
                            f"{stores} точек ({stores_pct:.0f}%)",
                            delta=f"{revenue_pct:.1f}% выручки",
                            help=f"Выручка: {format_number(revenue)} ₴"
                        )
                else:
                    with col3:
                        st.metric(
                            f"Категория {category}",
                            f"{stores} точек ({stores_pct:.0f}%)",
                            delta=f"{revenue_pct:.1f}% выручки",
                            help=f"Выручка: {format_number(revenue)} ₴"
                        )

        st.markdown("---")

        # Визуализация ABC
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Распределение выручки по категориям")

            fig_abc_pie = px.pie(
                abc_summary,
                values='Revenue',
                names='Category',
                title='Вклад категорий в общую выручку',
                color='Category',
                color_discrete_map={
                    'A': '#51cf66', 'B': '#ffd43b', 'C': '#ff6b6b'}
            )
            fig_abc_pie.update_traces(
                textposition='inside',
                textinfo='percent+label')
            fig_abc_pie.update_layout(height=350)
            st.plotly_chart(fig_abc_pie, use_container_width=True)

        with col2:
            st.subheader("Кривая Парето")

            # Парето кривая
            abc_pareto = abc_analysis[['Magazin',
                                       'Revenue_Fact',
                                       'Revenue_Cumsum_Pct',
                                       'ABC_Category']].head(20)

            fig_pareto = go.Figure()

            # Столбцы выручки
            fig_pareto.add_trace(go.Bar(
                x=abc_pareto['Magazin'],
                y=abc_pareto['Revenue_Fact'],
                name='Выручка',
                marker_color='lightblue',
                yaxis='y'
            ))

            # Линия накопительного процента
            fig_pareto.add_trace(go.Scatter(
                x=abc_pareto['Magazin'],
                y=abc_pareto['Revenue_Cumsum_Pct'],
                name='Накопительный %',
                line=dict(color='red', width=2),
                yaxis='y2'
            ))

            fig_pareto.update_layout(
                title='Парето анализ (топ-20 точек)',
                xaxis_title='Магазин',
                yaxis=dict(title='Выручка (₴)'),
                yaxis2=dict(
                    title='Накопительный %',
                    overlaying='y',
                    side='right'),
                hovermode='x unified',
                height=350,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1)
            )

            st.plotly_chart(fig_pareto, use_container_width=True)

        st.markdown("---")

        # Детальная таблица ABC
        st.subheader("📋 Детальная таблица ABC анализа")

        # Фильтр по категориям
        selected_abc = st.multiselect(
            "Выберите категории для отображения",
            options=['A', 'B', 'C'],
            default=['A', 'B', 'C']
        )

        abc_filtered = abc_analysis[abc_analysis['ABC_Category'].isin(
            selected_abc)].copy()

        # Форматирование таблицы
        abc_display = abc_filtered[['Magazin',
                                    'ABC_Category',
                                    'Revenue_Plan',
                                    'Revenue_Fact',
                                    'Performance',
                                    'Revenue_Cumsum_Pct']].copy()
        abc_display.columns = [
            'Магазин',
            'ABC',
            'План ₴',
            'Факт ₴',
            'Выполнение %',
            'Накопительно %']

        # Цветовое кодирование
        def color_abc(row):
            colors = {
                'A': 'background-color: #d3f9d8',
                'B': 'background-color: #fff3bf',
                'C': 'background-color: #ffe0e0'
            }
            return [colors.get(row['ABC'], '')] * len(row)

        styled_abc = abc_display.style.apply(color_abc, axis=1).format({
            'План ₴': lambda x: f"{format_number(x)} ₴",
            'Факт ₴': lambda x: f"{format_number(x)} ₴",
            'Выполнение %': lambda x: f"{x:.1f}%",
            'Накопительно %': lambda x: f"{x:.1f}%"
        })

        st.dataframe(styled_abc, use_container_width=True, height=400)

        # Рекомендации по категориям
        st.markdown("---")
        st.subheader("💡 Рекомендации по управлению")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Категория A (VIP)**")
            st.markdown("- ✅ Приоритетное внимание")
            st.markdown("- ✅ Персональные менеджеры")
            st.markdown("- ✅ Эксклюзивные условия")
            st.markdown("- ✅ Регулярный мониторинг")

        with col2:
            st.markdown("**Категория B (Стандарт)**")
            st.markdown("- 🔶 Стандартное обслуживание")
            st.markdown("- 🔶 Развитие потенциала")
            st.markdown("- 🔶 Стимулирующие программы")
            st.markdown("- 🔶 Периодический контроль")

        with col3:
            st.markdown("**Категория C (Проблемные)**")
            st.markdown("- ⚠️ Анализ причин низких продаж")
            st.markdown("- ⚠️ Оптимизация или закрытие")
            st.markdown("- ⚠️ Базовое обслуживание")
            st.markdown("- ⚠️ Минимальные инвестиции")

        # Экспорт данных
        st.markdown("---")
        st.subheader("💾 Экспорт данных")

        col1, col2 = st.columns(2)

        with col1:
            # Экспорт ABC анализа
            csv_abc = abc_display.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 Скачать ABC анализ (CSV)",
                data=csv_abc,
                file_name=f"abc_analysis_{min(selected_months)}_{max(selected_months)}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col2:
            # Экспорт Executive Summary
            summary_data = pd.DataFrame([{
                'Метрика': 'План выручки',
                'Значение': f"{format_number(financial_metrics['total_revenue_plan'])} ₴"
            }, {
                'Метрика': 'Факт выручки',
                'Значение': f"{format_number(financial_metrics['total_revenue_fact'])} ₴"
            }, {
                'Метрика': 'Выполнение плана',
                'Значение': f"{financial_metrics['plan_achievement']:.1f}%"
            }, {
                'Метрика': 'Средний чек',
                'Значение': f"{format_number(financial_metrics['avg_check_fact'])} ₴"
            }, {
                'Метрика': 'Успешность точек',
                'Значение': f"{financial_metrics['store_success_rate']:.0f}%"
            }])

            csv_summary = summary_data.to_csv(
                index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 Скачать Executive Summary (CSV)",
                data=csv_summary,
                file_name=f"executive_summary_{min(selected_months)}_{max(selected_months)}.csv",
                mime="text/csv",
                use_container_width=True
            )

    # TAB 6: Общая сводка анализа
    with tab6:
        st.header("📊 Общая сводка анализа План/Факт")
        st.markdown("---")

        st.info("Этот раздел предоставляет комплексный анализ выполнения плана продаж с детализацией по ключевым метрикам")

        # Период анализа
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader(f"📅 Период анализа: {min(selected_months)} - {max(selected_months)}")
        with col2:
            st.metric(
                "Количество месяцев",
                len(selected_months),
                help="Количество анализируемых месяцев"
            )

        st.markdown("---")

        # Сводная таблица по всем метрикам
        st.markdown("### 📈 Сводная таблица показателей")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### Выручка")
            revenue_summary = pd.DataFrame({
                'Показатель': ['План', 'Факт', 'Отклонение', 'Отклонение %'],
                'Значение': [
                    f"{format_number(financial_metrics['total_revenue_plan'])} ₴",
                    f"{format_number(financial_metrics['total_revenue_fact'])} ₴",
                    f"{format_number(financial_metrics['revenue_variance'])} ₴",
                    f"{financial_metrics['revenue_variance_pct']:+.2f}%"
                ]
            })
            st.dataframe(revenue_summary, use_container_width=True, hide_index=True)

        with col2:
            st.markdown("#### Количество продаж")
            units_summary = pd.DataFrame({
                'Показатель': ['План', 'Факт', 'Отклонение', 'Отклонение %'],
                'Значение': [
                    f"{format_number(financial_metrics['total_units_plan'])} шт",
                    f"{format_number(financial_metrics['total_units_fact'])} шт",
                    f"{format_number(financial_metrics['units_variance'])} шт",
                    f"{financial_metrics['units_variance_pct']:+.2f}%"
                ]
            })
            st.dataframe(units_summary, use_container_width=True, hide_index=True)

        with col3:
            st.markdown("#### Средний чек")
            avg_check_summary = pd.DataFrame({
                'Показатель': ['План', 'Факт', 'Отклонение', 'Отклонение %'],
                'Значение': [
                    f"{format_number(financial_metrics['avg_check_plan'])} ₴",
                    f"{format_number(financial_metrics['avg_check_fact'])} ₴",
                    f"{format_number(financial_metrics['avg_check_diff'])} ₴",
                    f"{financial_metrics['avg_check_diff_pct']:+.2f}%"
                ]
            })
            st.dataframe(avg_check_summary, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Детализация по сегментам
        st.markdown("### 📦 Детальный анализ по сегментам")

        segment_detailed = df_filtered.groupby('Segment').agg({
            'Revenue_Plan': 'sum',
            'Revenue_Fact': 'sum',
            'Units_Plan': 'sum',
            'Units_Fact': 'sum'
        }).reset_index()

        segment_detailed['Revenue_Diff'] = segment_detailed['Revenue_Fact'] - segment_detailed['Revenue_Plan']
        segment_detailed['Revenue_Diff_Pct'] = safe_divide(
            segment_detailed['Revenue_Diff'],
            segment_detailed['Revenue_Plan']
        ) * 100

        segment_detailed['Units_Diff'] = segment_detailed['Units_Fact'] - segment_detailed['Units_Plan']
        segment_detailed['Units_Diff_Pct'] = safe_divide(
            segment_detailed['Units_Diff'],
            segment_detailed['Units_Plan']
        ) * 100

        segment_detailed['Avg_Check_Fact'] = safe_divide(
            segment_detailed['Revenue_Fact'],
            segment_detailed['Units_Fact']
        )

        # Форматирование
        segment_display = segment_detailed.copy()
        segment_display['Revenue_Plan'] = segment_display['Revenue_Plan'].apply(lambda x: f"{format_number(x)} ₴")
        segment_display['Revenue_Fact'] = segment_display['Revenue_Fact'].apply(lambda x: f"{format_number(x)} ₴")
        segment_display['Revenue_Diff'] = segment_display['Revenue_Diff'].apply(lambda x: f"{format_number(x)} ₴")
        segment_display['Revenue_Diff_Pct'] = segment_display['Revenue_Diff_Pct'].apply(lambda x: f"{x:+.1f}%")
        segment_display['Units_Plan'] = segment_display['Units_Plan'].apply(lambda x: f"{format_number(x)} шт")
        segment_display['Units_Fact'] = segment_display['Units_Fact'].apply(lambda x: f"{format_number(x)} шт")
        segment_display['Units_Diff'] = segment_display['Units_Diff'].apply(lambda x: f"{format_number(x)} шт")
        segment_display['Units_Diff_Pct'] = segment_display['Units_Diff_Pct'].apply(lambda x: f"{x:+.1f}%")
        segment_display['Avg_Check_Fact'] = segment_display['Avg_Check_Fact'].apply(lambda x: f"{format_number(x)} ₴")

        st.dataframe(segment_display, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Детализация по магазинам - ТОП и ХУДШИЕ
        st.markdown("### 🏪 Лучшие и худшие магазины по выполнению плана")

        col1, col2 = st.columns(2)

        store_performance = df_filtered.groupby('Magazin').agg({
            'Revenue_Plan': 'sum',
            'Revenue_Fact': 'sum'
        }).reset_index()

        store_performance['Achievement_%'] = safe_divide(
            store_performance['Revenue_Fact'],
            store_performance['Revenue_Plan']
        ) * 100

        store_performance_sorted = store_performance.sort_values('Achievement_%', ascending=False)

        with col1:
            st.markdown("#### 🟢 ТОП-10 магазинов")
            top10 = store_performance_sorted.head(10).copy()
            top10['Revenue_Plan'] = top10['Revenue_Plan'].apply(lambda x: f"{format_number(x)} ₴")
            top10['Revenue_Fact'] = top10['Revenue_Fact'].apply(lambda x: f"{format_number(x)} ₴")
            top10['Achievement_%'] = top10['Achievement_%'].apply(lambda x: f"{x:.1f}%")
            st.dataframe(top10, use_container_width=True, hide_index=True, height=400)

        with col2:
            st.markdown("#### 🔴 ХУДШИЕ-10 магазинов")
            bottom10 = store_performance_sorted.tail(10).copy()
            bottom10['Revenue_Plan'] = bottom10['Revenue_Plan'].apply(lambda x: f"{format_number(x)} ₴")
            bottom10['Revenue_Fact'] = bottom10['Revenue_Fact'].apply(lambda x: f"{format_number(x)} ₴")
            bottom10['Achievement_%'] = bottom10['Achievement_%'].apply(lambda x: f"{x:.1f}%")
            st.dataframe(bottom10, use_container_width=True, hide_index=True, height=400)

        st.markdown("---")

        # График распределения выполнения плана
        st.markdown("### 📊 Распределение магазинов по выполнению плана")

        fig_distribution = px.histogram(
            store_performance,
            x='Achievement_%',
            nbins=20,
            title='Количество магазинов по уровню выполнения плана',
            labels={'Achievement_%': 'Выполнение плана (%)', 'count': 'Количество магазинов'},
            color_discrete_sequence=['#4dabf7']
        )

        # Добавляем линию на уровне 100%
        fig_distribution.add_vline(
            x=100,
            line_dash="dash",
            line_color="red",
            annotation_text="100% плана",
            annotation_position="top"
        )

        fig_distribution.update_layout(height=400)
        st.plotly_chart(fig_distribution, use_container_width=True)

        # Экспорт общей сводки
        st.markdown("---")
        st.markdown("### 💾 Экспорт данных")

        col1, col2 = st.columns(2)

        with col1:
            # Экспорт сводки по сегментам
            csv_segment = segment_display.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 Скачать анализ по сегментам (CSV)",
                data=csv_segment,
                file_name=f"segment_analysis_{min(selected_months)}_{max(selected_months)}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col2:
            # Экспорт по магазинам
            csv_stores = store_performance.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 Скачать анализ по магазинам (CSV)",
                data=csv_stores,
                file_name=f"store_analysis_{min(selected_months)}_{max(selected_months)}.csv",
                mime="text/csv",
                use_container_width=True
            )

    # TAB 7: Планирование
    with tab7:
        st.header("🎯 Интеллектуальное планирование продаж")
        st.markdown("---")

        st.info("Этот раздел использует исторические данные и машинное обучение для прогнозирования будущих продаж и генерации рекомендаций")

        # Параметры прогнозирования
        col1, col2, col3 = st.columns(3)

        with col1:
            forecast_periods = st.slider(
                "Количество месяцев для прогноза",
                min_value=1,
                max_value=12,
                value=3,
                help="Выберите количество месяцев для прогнозирования"
            )

        with col2:
            adjustment_factor = st.slider(
                "Коэффициент корректировки плана",
                min_value=0.5,
                max_value=1.5,
                value=1.0,
                step=0.05,
                help="1.0 = без изменений, 1.1 = +10%, 0.9 = -10%"
            )

        with col3:
            st.metric(
                "Корректировка",
                f"{(adjustment_factor - 1) * 100:+.0f}%",
                help="Процент корректировки от базового прогноза"
            )

        st.markdown("---")

        # Анализ тренда
        st.markdown("### 📈 Анализ текущего тренда продаж")

        avg_growth_rate, monthly_sales = calculate_growth_rate(df_filtered, df_fact_detailed)

        col1, col2, col3 = st.columns(3)

        with col1:
            growth_color = "normal" if avg_growth_rate >= 0 else "inverse"
            st.metric(
                "Средний темп роста",
                f"{avg_growth_rate:+.2f}%",
                help="Средний месячный темп роста продаж",
                delta_color=growth_color
            )

        with col2:
            if not monthly_sales.empty:
                last_month_revenue = monthly_sales.iloc[-1]['Sum']
                st.metric(
                    "Выручка последнего месяца",
                    f"{format_number(last_month_revenue)} ₴",
                    help="Выручка за последний полный месяц"
                )

        with col3:
            if len(monthly_sales) >= 2:
                mom_growth = ((monthly_sales.iloc[-1]['Sum'] - monthly_sales.iloc[-2]['Sum']) /
                             monthly_sales.iloc[-2]['Sum'] * 100) if monthly_sales.iloc[-2]['Sum'] > 0 else 0
                st.metric(
                    "Рост к предыдущему месяцу",
                    f"{mom_growth:+.1f}%",
                    help="Month-over-Month рост"
                )

        # График исторических продаж
        if not monthly_sales.empty:
            fig_trend = px.line(
                monthly_sales,
                x='Month',
                y='Sum',
                markers=True,
                title='Тренд продаж по месяцам',
                labels={'Sum': 'Выручка (₴)', 'Month': 'Месяц'}
            )
            fig_trend.update_layout(height=350)
            st.plotly_chart(fig_trend, use_container_width=True)

        st.markdown("---")

        # Прогноз с моделями машинного обучения
        st.markdown("### 🔮 Прогноз продаж с ML моделями")

        st.info("Система использует 5 различных моделей машинного обучения для максимально точного прогнозирования")

        # Выбор модели и сценария
        col1, col2 = st.columns(2)

        with col1:
            selected_model = st.selectbox(
                "📊 Выберите модель прогнозирования",
                options=[
                    'ensemble',
                    'linear',
                    'polynomial',
                    'exponential',
                    'wma'
                ],
                format_func=lambda x: {
                    'ensemble': '🏆 Ансамбль моделей (рекомендуется)',
                    'linear': '📈 Линейная регрессия',
                    'polynomial': '📊 Полиномиальная регрессия',
                    'exponential': '📉 Экспоненциальное сглаживание',
                    'wma': '📐 Взвешенное скользящее среднее'
                }[x],
                help="Ансамбль моделей объединяет все модели для максимальной точности"
            )

        with col2:
            selected_scenario = st.selectbox(
                "🎬 Выберите сценарий планирования",
                options=['optimistic', 'realistic', 'pessimistic'],
                format_func=lambda x: {
                    'optimistic': '🟢 Оптимистичный (+20%)',
                    'realistic': '🟡 Реальный (базовый)',
                    'pessimistic': '🔴 Пессимистичный (-15%)'
                }[x],
                index=1,  # По умолчанию реальный
                help="Сценарий применяет коэффициент к базовому прогнозу"
            )

        if st.button("🚀 Сгенерировать прогноз", type="primary", use_container_width=False):
            with st.spinner("Анализ данных и генерация прогноза с использованием ML..."):
                # Генерируем прогнозы для всех моделей
                all_forecasts_df = forecast_with_multiple_models(df_filtered, df_fact_detailed, forecast_periods)

                if all_forecasts_df is not None and not all_forecasts_df.empty:
                    st.success("✅ Прогноз успешно сгенерирован с использованием машинного обучения")

                    # Фильтруем по выбранной модели
                    selected_forecast = all_forecasts_df[all_forecasts_df['Model_Key'] == selected_model].copy()

                    # Применяем сценарий
                    scenario_forecast = apply_scenario(selected_forecast, selected_scenario)

                    # Показываем точность модели
                    st.markdown("### 🎯 Точность выбранной модели")

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        model_name = selected_forecast['Model'].iloc[0]
                        mean_value = selected_forecast['Mean_Value'].iloc[0]
                        st.info(f"**Модель:** {model_name}")
                        st.caption(f"Средняя выручка: {format_number(mean_value)} ₴")

                    with col2:
                        mape = selected_forecast['MAPE'].iloc[0]
                        mape_color = "🟢" if mape < 10 else "🟡" if mape < 20 else "🔴"
                        st.metric(
                            "MAPE",
                            f"{mape:.2f}%",
                            help="Mean Absolute Percentage Error - средняя процентная ошибка. < 10% - отлично, < 20% - хорошо"
                        )
                        st.caption(f"{mape_color} Точность")

                    with col3:
                        rmse_pct = selected_forecast['RMSE_Pct'].iloc[0]
                        rmse_color = "🟢" if rmse_pct < 10 else "🟡" if rmse_pct < 20 else "🔴"
                        st.metric(
                            "RMSE",
                            f"{rmse_pct:.2f}%",
                            help="Root Mean Squared Error - среднеквадратичная ошибка в процентах от средней выручки"
                        )
                        st.caption(f"{rmse_color} Отклонение")

                    with col4:
                        mae_pct = selected_forecast['MAE_Pct'].iloc[0]
                        mae_color = "🟢" if mae_pct < 10 else "🟡" if mae_pct < 20 else "🔴"
                        st.metric(
                            "MAE",
                            f"{mae_pct:.2f}%",
                            help="Mean Absolute Error - средняя абсолютная ошибка в процентах от средней выручки"
                        )
                        st.caption(f"{mae_color} Отклонение")

                    st.markdown("---")

                    # Сравнение всех моделей
                    with st.expander("📊 Сравнение всех моделей"):
                        st.markdown("#### Точность моделей на исторических данных")

                        # Группируем по модели и берем уникальные значения метрик
                        model_comparison = all_forecasts_df.groupby('Model').agg({
                            'MAPE': 'first',
                            'RMSE_Pct': 'first',
                            'MAE_Pct': 'first',
                            'Mean_Value': 'first'
                        }).reset_index()

                        # Сортируем по MAPE (лучшие модели сверху)
                        model_comparison = model_comparison.sort_values('MAPE')

                        # Форматирование
                        model_comparison_display = model_comparison.copy()
                        model_comparison_display['MAPE'] = model_comparison_display['MAPE'].apply(lambda x: f"{x:.2f}%")
                        model_comparison_display['RMSE_Pct'] = model_comparison_display['RMSE_Pct'].apply(lambda x: f"{x:.2f}%")
                        model_comparison_display['MAE_Pct'] = model_comparison_display['MAE_Pct'].apply(lambda x: f"{x:.2f}%")

                        # Добавляем цветовые индикаторы
                        def get_accuracy_indicator(val_str):
                            val = float(val_str.replace('%', ''))
                            if val < 10:
                                return f"🟢 {val_str}"
                            elif val < 20:
                                return f"🟡 {val_str}"
                            else:
                                return f"🔴 {val_str}"

                        model_comparison_display['MAPE'] = model_comparison_display['MAPE'].apply(get_accuracy_indicator)
                        model_comparison_display['RMSE_Pct'] = model_comparison_display['RMSE_Pct'].apply(get_accuracy_indicator)
                        model_comparison_display['MAE_Pct'] = model_comparison_display['MAE_Pct'].apply(get_accuracy_indicator)

                        model_comparison_display = model_comparison_display[['Model', 'MAPE', 'RMSE_Pct', 'MAE_Pct']]
                        model_comparison_display.columns = ['Модель', 'MAPE', 'RMSE', 'MAE']

                        st.dataframe(model_comparison_display, use_container_width=True, hide_index=True)

                        st.caption("💡 Меньшие значения = более точная модель. 🟢 < 10% отлично | 🟡 10-20% хорошо | 🔴 > 20% требует улучшения")

                    st.markdown("---")

                    # Отображение прогноза по сценарию
                    st.markdown(f"### 📊 Прогноз: {scenario_forecast['Scenario'].iloc[0]}")

                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.markdown("#### 📅 Прогнозные показатели")

                        forecast_display = scenario_forecast[['Month', 'Forecast_Revenue', 'Forecast_Units', 'Scenario']].copy()
                        forecast_display['Forecast_Revenue'] = forecast_display['Forecast_Revenue'].apply(
                            lambda x: f"{format_number(x)} ₴")
                        forecast_display['Forecast_Units'] = forecast_display['Forecast_Units'].apply(
                            lambda x: f"{format_number(x)} шт")

                        forecast_display.columns = ['Месяц', 'Прогноз выручки', 'Прогноз штук', 'Сценарий']
                        st.dataframe(forecast_display, use_container_width=True, hide_index=True)

                    with col2:
                        st.markdown("#### 💰 Итого прогноз")
                        total_forecast_revenue = scenario_forecast['Forecast_Revenue'].sum()
                        total_forecast_units = scenario_forecast['Forecast_Units'].sum()

                        st.metric("Выручка", f"{format_number(total_forecast_revenue)} ₴")
                        st.metric("Количество", f"{format_number(total_forecast_units)} шт")
                        st.metric("Средний чек", f"{format_number(safe_divide(total_forecast_revenue, total_forecast_units))} ₴")

                    st.markdown("---")

                    # Сравнение сценариев
                    st.markdown("### 🎬 Сравнение сценариев")

                    # Получаем базовый прогноз (без сценария)
                    base_forecast = all_forecasts_df[all_forecasts_df['Model_Key'] == selected_model].copy()

                    # Убедимся, что есть данные для прогноза
                    if base_forecast.empty:
                        st.warning("Нет данных для построения сценариев")
                    else:
                        # Генерируем прогнозы для всех трех сценариев явно (без цикла)
                        # Это гарантирует, что каждый сценарий обрабатывается независимо
                        scenario_optimistic = apply_scenario(base_forecast.copy(), 'optimistic')
                        scenario_realistic = apply_scenario(base_forecast.copy(), 'realistic')
                        scenario_pessimistic = apply_scenario(base_forecast.copy(), 'pessimistic')

                        # Объединяем все сценарии
                        all_scenarios_df = pd.concat([
                            scenario_optimistic,
                            scenario_realistic,
                            scenario_pessimistic
                        ], ignore_index=True)

                        # Подготавливаем данные для графика
                        # Выбираем только нужные колонки и группируем по месяцу и сценарию
                        scenario_chart = all_scenarios_df[['Month', 'Scenario', 'Forecast_Revenue', 'Scenario_Factor']].copy()

                        # Группируем по месяцу и сценарию (на случай если есть дубликаты)
                        scenario_chart = scenario_chart.groupby(['Month', 'Scenario', 'Scenario_Factor'], as_index=False).agg({
                            'Forecast_Revenue': 'sum'
                        })

                        # Сортируем по месяцу для правильного отображения линий
                        scenario_chart = scenario_chart.sort_values(['Month', 'Scenario'])

                        # Отладочная информация (скрыта по умолчанию)
                        with st.expander("🔍 Проверка данных (отладка)"):
                            st.write("**Базовый прогноз:**")
                            st.write(f"Количество строк: {len(base_forecast)}")
                            if len(base_forecast) > 0:
                                st.write(f"Первый месяц: {base_forecast['Month'].iloc[0]}")
                                st.write(f"Выручка первого месяца: {base_forecast['Forecast_Revenue'].iloc[0]:,.2f} ₴")

                            st.write("\n**После применения сценариев (все месяцы):**")

                            for scenario_name in ['Оптимистичный', 'Реальный', 'Пессимистичный']:
                                scenario_rows = all_scenarios_df[all_scenarios_df['Scenario'] == scenario_name]
                                if not scenario_rows.empty:
                                    factor = scenario_rows['Scenario_Factor'].iloc[0]
                                    total_revenue = scenario_rows['Forecast_Revenue'].sum()
                                    st.write(f"\n{scenario_name} (фактор {factor}):")
                                    st.write(f"  Общая выручка: {total_revenue:,.2f} ₴")
                                    for idx, row in scenario_rows.iterrows():
                                        st.write(f"  {row['Month']}: {row['Forecast_Revenue']:,.2f} ₴")

                            st.write(f"\n**Количество точек данных в графике:** {len(scenario_chart)}")
                            st.write(f"**Уникальные сценарии:** {scenario_chart['Scenario'].unique().tolist()}")

                            st.write("\n**Данные для графика:**")
                            st.dataframe(scenario_chart)

                        # График сравнения сценариев
                        fig_scenarios = px.line(
                            scenario_chart,
                            x='Month',
                            y='Forecast_Revenue',
                            color='Scenario',
                            markers=True,
                            title='Сравнение сценариев прогноза',
                            labels={'Forecast_Revenue': 'Выручка (₴)', 'Month': 'Месяц', 'Scenario': 'Сценарий'},
                            color_discrete_map={
                                'Оптимистичный': '#51cf66',
                                'Реальный': '#4dabf7',
                                'Пессимистичный': '#ff6b6b'
                            }
                        )
                        fig_scenarios.update_layout(height=400)
                        st.plotly_chart(fig_scenarios, use_container_width=True)

                        # Таблица сравнения сценариев
                        st.markdown("#### 📊 Сводка по сценариям")

                        scenarios_summary = all_scenarios_df.groupby('Scenario').agg({
                            'Forecast_Revenue': 'sum',
                            'Forecast_Units': 'sum',
                            'Scenario_Factor': 'first'
                        }).reset_index()

                        scenarios_summary['Avg_Check'] = safe_divide(
                            scenarios_summary['Forecast_Revenue'],
                            scenarios_summary['Forecast_Units']
                        )

                        # Сортируем в нужном порядке
                        scenario_order = {'Оптимистичный': 0, 'Реальный': 1, 'Пессимистичный': 2}
                        scenarios_summary['Sort_Order'] = scenarios_summary['Scenario'].map(scenario_order)
                        scenarios_summary = scenarios_summary.sort_values('Sort_Order').drop('Sort_Order', axis=1)

                        scenarios_summary_display = scenarios_summary.copy()
                        scenarios_summary_display['Коэффициент'] = scenarios_summary_display['Scenario_Factor'].apply(
                            lambda x: f"×{x:.2f} ({(x-1)*100:+.0f}%)")
                        scenarios_summary_display['Forecast_Revenue'] = scenarios_summary_display['Forecast_Revenue'].apply(
                            lambda x: f"{format_number(x)} ₴")
                        scenarios_summary_display['Forecast_Units'] = scenarios_summary_display['Forecast_Units'].apply(
                            lambda x: f"{format_number(x)} шт")
                        scenarios_summary_display['Avg_Check'] = scenarios_summary_display['Avg_Check'].apply(
                            lambda x: f"{format_number(x)} ₴")

                        scenarios_summary_display = scenarios_summary_display[['Scenario', 'Коэффициент', 'Forecast_Revenue', 'Forecast_Units', 'Avg_Check']]
                        scenarios_summary_display.columns = ['Сценарий', 'Коэффициент', 'Прогноз выручки', 'Прогноз штук', 'Средний чек']
                        st.dataframe(scenarios_summary_display, use_container_width=True, hide_index=True)

                    # График прогноза с историческими данными
                    st.markdown("---")
                    st.markdown("### 📈 Прогноз на основе исторических данных")

                    combined_data = monthly_sales.copy()
                    combined_data['Type'] = 'Факт'
                    combined_data = combined_data.rename(columns={'Sum': 'Revenue'})
                    combined_data = combined_data[['Month', 'Revenue', 'Type']]

                    forecast_chart = scenario_forecast[['Month', 'Forecast_Revenue', 'Scenario']].copy()
                    forecast_chart = forecast_chart.rename(columns={'Forecast_Revenue': 'Revenue', 'Scenario': 'Type'})

                    combined_chart = pd.concat([combined_data, forecast_chart[['Month', 'Revenue', 'Type']]])

                    fig_forecast = px.line(
                        combined_chart,
                        x='Month',
                        y='Revenue',
                        color='Type',
                        markers=True,
                        title=f'Исторические данные и прогноз ({scenario_forecast["Scenario"].iloc[0]})',
                        labels={'Revenue': 'Выручка (₴)', 'Month': 'Месяц', 'Type': 'Тип'},
                        color_discrete_map={
                            'Факт': '#4dabf7',
                            'Оптимистичный': '#51cf66',
                            'Реальный': '#ffd43b',
                            'Пессимистичный': '#ff6b6b'
                        }
                    )
                    fig_forecast.update_layout(height=400)
                    st.plotly_chart(fig_forecast, use_container_width=True)

                    # Экспорт прогноза
                    st.markdown("---")
                    csv_forecast = scenario_forecast.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="📥 Скачать прогноз (CSV)",
                        data=csv_forecast,
                        file_name=f"forecast_{selected_model}_{selected_scenario}_{forecast_periods}months.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                else:
                    st.error("❌ Недостаточно данных для генерации прогноза")

        st.markdown("---")

        # Генерация умного плана
        st.markdown("### 🧠 Генерация умного плана")

        st.info("Система автоматически создаст план продаж на основе исторических данных, текущих трендов и выбранного коэффициента корректировки")

        if st.button("🎯 Создать умный план", type="primary", use_container_width=False):
            with st.spinner("Генерация плана..."):
                smart_plan_df = create_smart_plan(
                    df_filtered,
                    df_fact_detailed,
                    forecast_periods,
                    adjustment_factor
                )

                if smart_plan_df is not None and not smart_plan_df.empty:
                    st.success(f"✅ План успешно создан для {len(smart_plan_df)} позиций")

                    # Группировка по месяцам
                    plan_by_month = smart_plan_df.groupby('Month').agg({
                        'Revenue_Plan': 'sum',
                        'Units_Plan': 'sum'
                    }).reset_index()

                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.markdown("#### 📅 План по месяцам")
                        plan_display = plan_by_month.copy()
                        plan_display['Revenue_Plan'] = plan_display['Revenue_Plan'].apply(
                            lambda x: f"{format_number(x)} ₴")
                        plan_display['Units_Plan'] = plan_display['Units_Plan'].apply(
                            lambda x: f"{format_number(x)} шт")
                        plan_display.columns = ['Месяц', 'План выручки', 'План штук']
                        st.dataframe(plan_display, use_container_width=True, hide_index=True)

                    with col2:
                        st.markdown("#### 💼 Итого план")
                        total_plan_revenue = plan_by_month['Revenue_Plan'].sum()
                        total_plan_units = plan_by_month['Units_Plan'].sum()

                        st.metric("План выручки", f"{format_number(total_plan_revenue)} ₴")
                        st.metric("План штук", f"{format_number(total_plan_units)} шт")
                        st.metric("Средний чек", f"{format_number(safe_divide(total_plan_revenue, total_plan_units))} ₴")

                    # Детальный план
                    st.markdown("---")
                    st.markdown("#### 📋 Детальный план по магазинам и сегментам")

                    # Опция фильтрации
                    selected_plan_month = st.selectbox(
                        "Выберите месяц для просмотра",
                        options=sorted(smart_plan_df['Month'].unique())
                    )

                    plan_filtered = smart_plan_df[smart_plan_df['Month'] == selected_plan_month].copy()
                    plan_filtered['Revenue_Plan'] = plan_filtered['Revenue_Plan'].apply(
                        lambda x: f"{format_number(x)} ₴")
                    plan_filtered['Units_Plan'] = plan_filtered['Units_Plan'].apply(
                        lambda x: f"{format_number(x)} шт")
                    plan_filtered['Growth_Rate'] = plan_filtered['Growth_Rate'].apply(
                        lambda x: f"{x:.2f}%")

                    st.dataframe(plan_filtered, use_container_width=True, hide_index=True, height=400)

                    # Экспорт плана
                    st.markdown("---")
                    csv_plan = smart_plan_df.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="📥 Скачать сгенерированный план (CSV)",
                        data=csv_plan,
                        file_name=f"smart_plan_{forecast_periods}months.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                else:
                    st.error("❌ Недостаточно данных для генерации плана")

        st.markdown("---")

        # Рекомендации
        st.markdown("### 💡 Рекомендации для планирования")

        recommendations_df = generate_plan_recommendations(
            df_filtered,
            df_fact_detailed,
            financial_metrics,
            abc_analysis
        )

        if recommendations_df is not None and not recommendations_df.empty:
            st.success(f"Сгенерировано {len(recommendations_df)} рекомендаций")

            # Фильтр по приоритету
            priority_filter = st.multiselect(
                "Фильтр по приоритету",
                options=recommendations_df['priority'].unique(),
                default=recommendations_df['priority'].unique()
            )

            recommendations_filtered = recommendations_df[recommendations_df['priority'].isin(priority_filter)]

            # Цветовое кодирование по приоритету
            def color_priority(row):
                colors = {
                    'Критический': 'background-color: #ffe0e0',
                    'Высокий': 'background-color: #fff3bf',
                    'Средний': 'background-color: #e3f2fd',
                }
                return [colors.get(row['priority'], '')] * len(row)

            styled_recommendations = recommendations_filtered.style.apply(color_priority, axis=1)

            st.dataframe(styled_recommendations, use_container_width=True, height=400)

            # Экспорт рекомендаций
            csv_recommendations = recommendations_filtered.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 Скачать рекомендации (CSV)",
                data=csv_recommendations,
                file_name=f"recommendations_{min(selected_months)}_{max(selected_months)}.csv",
                mime="text/csv",
                use_container_width=True
            )

        else:
            st.info("📊 Все показатели в пределах нормы. Специальных рекомендаций нет.")

        st.markdown("---")

        # Методология
        with st.expander("ℹ️ Методология расчетов и ML моделей"):
            st.markdown("""
            ### Как работает система планирования

            **1. Анализ трендов:**
            - Расчет среднего темпа роста на основе исторических данных
            - Анализ сезонности и паттернов продаж
            - Оценка волатильности показателей

            **2. ML модели прогнозирования:**

            **Линейная регрессия:**
            - Находит линейный тренд в данных
            - Формула: `y = a × x + b`
            - Лучше для стабильного роста/падения

            **Полиномиальная регрессия (степень 2):**
            - Улавливает нелинейные паттерны
            - Формула: `y = a × x² + b × x + c`
            - Лучше для ускоряющихся/замедляющихся трендов

            **Экспоненциальное сглаживание (α=0.3):**
            - Взвешивает исторические данные
            - Формула: `S_t = α × Y_t + (1-α) × S_{t-1}`
            - Больший вес на свежие данные

            **Взвешенное скользящее среднее (окно=3):**
            - Приоритет последним N периодам
            - Веса: `[1, 2, 3] / 6` для окна=3
            - Адаптируется к изменениям

            **Ансамбль моделей (РЕКОМЕНДУЕТСЯ):**
            - Среднее всех моделей
            - Формула: `(Модель1 + Модель2 + ... + МодельN) / N`
            - Максимальная точность и надежность

            **3. Метрики точности прогноза:**

            **MAPE (Mean Absolute Percentage Error):**
            - Средняя процентная ошибка
            - Формула: `(|Факт - Прогноз| / Факт) × 100%`
            - 🟢 < 10% = отлично | 🟡 10-20% = хорошо | 🔴 > 20% = требует улучшения

            **RMSE (Root Mean Squared Error):**
            - Среднеквадратичная ошибка в % от средней выручки
            - Формула: `sqrt(mean((Факт - Прогноз)²)) / Средняя_выручка × 100%`
            - Показывает величину типичной ошибки
            - 🟢 < 10% = отлично | 🟡 10-20% = хорошо | 🔴 > 20% = требует улучшения

            **MAE (Mean Absolute Error):**
            - Средняя абсолютная ошибка в % от средней выручки
            - Формула: `mean(|Факт - Прогноз|) / Средняя_выручка × 100%`
            - Показывает среднее отклонение
            - 🟢 < 10% = отлично | 🟡 10-20% = хорошо | 🔴 > 20% = требует улучшения

            **4. Сценарии планирования:**
            - 🟢 Оптимистичный: прогноз × 1.20 (+20%)
            - 🟡 Реальный: прогноз × 1.00 (базовый)
            - 🔴 Пессимистичный: прогноз × 0.85 (-15%)

            **5. Генерация плана:**
            - Распределение общего прогноза по магазинам и сегментам
            - Учет исторической доли каждого магазина/сегмента
            - Применение сценарных коэффициентов

            **6. Рекомендации:**
            - Анализ отклонений от плана
            - Выявление проблемных зон
            - Генерация actionable рекомендаций

            **💡 Совет:** Используйте **Ансамбль моделей** с **Реальным сценарием** для наиболее точного и сбалансированного прогноза.
            """)


if __name__ == "__main__":
    main()
