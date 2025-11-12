import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
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

@st.cache_data(ttl=600)
def load_data_from_sheets(spreadsheet_id, plan_gid, fact_gid):
    """Загрузка из Google Sheets (публичный доступ)"""
    try:
        base_url = f'https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export'
        
        plan_url = f'{base_url}?format=csv&gid={plan_gid}'
        df_plan = pd.read_csv(plan_url)
        
        fact_url = f'{base_url}?format=csv&gid={fact_gid}'
        df_fact = pd.read_csv(fact_url)
        
        return df_fact, df_plan
    
    except Exception as e:
        st.error(f"Ошибка: {str(e)}")
        st.info("Проверь: таблица публична, GID правильные")
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
    df_fact['Datasales'] = pd.to_datetime(df_fact['Datasales'])
    df_fact['Month'] = df_fact['Datasales'].dt.to_period('M').astype(str)
    df_fact['Week'] = df_fact['Datasales'].dt.to_period('W').astype(str)
    
    # Агрегация факта по магазин × сегмент × месяц
    fact_agg = df_fact.groupby(['Magazin', 'Segment', 'Month']).agg({
        'Sum': 'sum',
        'Qty': 'sum'
    }).reset_index()
    fact_agg.columns = ['Magazin', 'Segment', 'Month', 'Revenue_Fact', 'Units_Fact']
    
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
    
    # Расчет отклонений
    df_merged['Revenue_Diff'] = df_merged['Revenue_Fact'] - df_merged['Revenue_Plan']
    df_merged['Revenue_Diff_Pct'] = (df_merged['Revenue_Diff'] / df_merged['Revenue_Plan'] * 100).round(2)
    
    df_merged['Units_Diff'] = df_merged['Units_Fact'] - df_merged['Units_Plan']
    df_merged['Units_Diff_Pct'] = (df_merged['Units_Diff'] / df_merged['Units_Plan'] * 100).round(2)
    
    return df_merged, df_fact

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
        
        spreadsheet_id = st.sidebar.text_input(
            "Spreadsheet ID",
            value="1lJLON5N_EKQ5ICv0Pprp5DamP1tNAhBIph4uEoWC04Q"
        )
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            plan_gid = st.text_input("Plan GID", value="103045414")
        with col2:
            fact_gid = st.text_input("Fact GID", value="1144131206")
        
        if st.sidebar.button("🔄 Загрузить данные"):
            with st.spinner("Загрузка..."):
                df_fact, df_plan = load_data_from_sheets(spreadsheet_id, plan_gid, fact_gid)
        else:
            st.info("👈 Нажми 'Загрузить данные' в боковой панели")
            return
    
    if df_fact is None or df_plan is None:
        st.warning("Данные не загружены. Используйте демо-данные или проверьте подключение.")
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
    
    # Основной контент - Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Сводка", "🏪 По магазинам", "📦 По сегментам", "📈 Динамика"])
    
    # TAB 1: Сводка
    with tab1:
        # KPI карточки
        col1, col2, col3, col4 = st.columns(4)
        
        total_revenue_plan = df_filtered['Revenue_Plan'].sum()
        total_revenue_fact = df_filtered['Revenue_Fact'].sum()
        total_revenue_diff = total_revenue_fact - total_revenue_plan
        total_revenue_diff_pct = (total_revenue_diff / total_revenue_plan * 100) if total_revenue_plan > 0 else 0
        
        total_units_plan = df_filtered['Units_Plan'].sum()
        total_units_fact = df_filtered['Units_Fact'].sum()
        total_units_diff = total_units_fact - total_units_plan
        total_units_diff_pct = (total_units_diff / total_units_plan * 100) if total_units_plan > 0 else 0
        
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
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_revenue, use_container_width=True)
        
        with col2:
            st.subheader("Выполнение плана по сегментам")
            
            perf_by_segment = df_filtered.groupby('Segment').agg({
                'Revenue_Plan': 'sum',
                'Revenue_Fact': 'sum'
            }).reset_index()
            perf_by_segment['Performance'] = (
                perf_by_segment['Revenue_Fact'] / perf_by_segment['Revenue_Plan'] * 100
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
            fig_segment.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_segment.update_layout(
                height=400,
                xaxis_title="Сегмент",
                yaxis_title="Выполнение плана (%)",
                showlegend=False
            )
            fig_segment.add_hline(y=100, line_dash="dash", line_color="gray", annotation_text="100%")
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
        
        store_summary['Revenue_Diff'] = store_summary['Revenue_Fact'] - store_summary['Revenue_Plan']
        store_summary['Revenue_Diff_Pct'] = (
            store_summary['Revenue_Diff'] / store_summary['Revenue_Plan'] * 100
        ).round(2)
        
        store_summary['Units_Diff'] = store_summary['Units_Fact'] - store_summary['Units_Plan']
        store_summary['Units_Diff_Pct'] = (
            store_summary['Units_Diff'] / store_summary['Units_Plan'] * 100
        ).round(2)
        
        # Форматирование таблицы
        def color_diff(val):
            if pd.isna(val):
                return ''
            color = 'green' if val >= 0 else 'red'
            return f'color: {color}'
        
        styled_table = store_summary.style.applymap(
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
            store_detail = df_filtered[df_filtered['Magazin'] == selected_store]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"Выполнение по сегментам: {selected_store}")
                
                fig_store_segment = px.bar(
                    store_detail,
                    x='Segment',
                    y=['Revenue_Plan', 'Revenue_Fact'],
                    barmode='group',
                    labels={'value': 'Выручка (₴)', 'variable': 'Тип'},
                    color_discrete_map={'Revenue_Plan': 'lightblue', 'Revenue_Fact': 'darkblue'}
                )
                fig_store_segment.update_layout(height=350)
                st.plotly_chart(fig_store_segment, use_container_width=True)
            
            with col2:
                st.subheader("Детали по сегментам")
                st.dataframe(
                    store_detail[['Segment', 'Month', 'Revenue_Plan', 'Revenue_Fact', 'Revenue_Diff_Pct']],
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
            segment_data = df_filtered[df_filtered['Segment'] == selected_segment]
            
            segment_by_store = segment_data.groupby('Magazin').agg({
                'Revenue_Plan': 'sum',
                'Revenue_Fact': 'sum',
                'Units_Plan': 'sum',
                'Units_Fact': 'sum'
            }).reset_index()
            
            segment_by_store['Revenue_Diff_Pct'] = (
                (segment_by_store['Revenue_Fact'] - segment_by_store['Revenue_Plan']) /
                segment_by_store['Revenue_Plan'] * 100
            ).round(2)
            
            # График топ/худшие магазины
            segment_by_store_sorted = segment_by_store.sort_values('Revenue_Diff_Pct')
            
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
                fig_top.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
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
                fig_bottom.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
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
            df_fact_detailed['Period'] = df_fact_detailed['Datasales'].dt.strftime('%Y-%m-%d')
        elif time_grain == 'Неделя':
            df_fact_detailed['Period'] = df_fact_detailed['Week']
        else:
            df_fact_detailed['Period'] = df_fact_detailed['Month']
        
        # Фильтр по выбранным месяцам
        df_fact_filtered = df_fact_detailed[df_fact_detailed['Month'].isin(selected_months)]
        
        daily_revenue = df_fact_filtered.groupby('Period')['Sum'].sum().reset_index()
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
        
        segment_timeline = df_fact_filtered.groupby(['Period', 'Segment'])['Sum'].sum().reset_index()
        
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
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_segment_timeline, use_container_width=True)

if __name__ == "__main__":
    main()