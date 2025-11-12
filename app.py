import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

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

        return df_fact, df_plan

    except Exception as e:
        st.error(f"Ошибка: {str(e)}")
        st.info("Проверь: таблица публична, ссылки правильные")
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

    # Преобразование дат
    df_fact['Datasales'] = pd.to_datetime(
        df_fact['Datasales'],
        format='%d.%m.%Y',
        errors='coerce')
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

# Главная функция


def main():

    # Заголовок
    st.title("👓 План/Факт Продаж Оптика")

    # Sidebar - фильтры
    st.sidebar.header("⚙️ Фильтры")

    # Загрузка данных
    use_demo = st.sidebar.checkbox("Использовать демо-данные", value=True)

    if use_demo:
        df_fact, df_plan = generate_demo_data()
    else:
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
    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Executive Summary",
        "📊 Сводка",
        "🏪 По магазинам",
        "📦 По сегментам",
        "📈 Динамика",
        "🎯 ABC Анализ"
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


if __name__ == "__main__":
    main()
