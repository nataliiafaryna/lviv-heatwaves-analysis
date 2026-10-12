from datetime import datetime
from itertools import count
import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymannkendall as mk
import matplotlib.dates as mdates
from scipy import stats
import matplotlib.patches as mpatches
from scipy.stats import alpha

#-------------------ПЕРШИЙ ЕТАП!!!!!!!!!!-----------------------

# Параметри для метеостанції Львів (Аеропорт): WMO ID = 33393
STATION_ID = "33393"
START_DATE = "1990-01-01"
END_DATE = "2025-12-31"

url = f"https://bulk.meteostat.net/v2/daily/{STATION_ID}.csv.gz"

try:
    df_raw = pd.read_csv(url, compression="gzip", header=None)

    # Структура колонок Meteostat v2 bulk CSV:
    # 0: date, 1: tavg, 2: tmin, 3: tmax, 4: prcp, ...
    df_raw.columns = [
        "date",
        "tavg",
        "tmin",
        "tmax",
        "prcp",
        "snow",
        "wdir",
        "wspd",
        "wpgt",
        "pres",
        "tsun",
    ]

    df_raw["date"] = pd.to_datetime(df_raw["date"])
    mask = (df_raw["date"] >= START_DATE) & (df_raw["date"] <= END_DATE)
    df_filtered = df_raw.loc[mask].copy()

    if not df_filtered.empty:
        df_clean = pd.DataFrame()
        df_clean["DATE"] = df_filtered["date"]
        df_clean["Tavg"] = df_filtered["tavg"].round(1)
        df_clean["Tmax"] = df_filtered["tmax"].round(1)
        df_clean["Tmin"] = df_filtered["tmin"].round(1)
        df_clean["PRCP"] = df_filtered["prcp"].fillna(0.0).round(1)

        # Захист від поодиноких пропусків у Tmax/Tmin
        df_clean["Tmax"] = df_clean["Tmax"].fillna(df_clean["Tavg"])
        df_clean["Tmin"] = df_clean["Tmin"].fillna(df_clean["Tavg"])

        # Збереження у CSV
        output_filename = "lviv_meteostat_station_2020_2025.csv"
        df_clean.to_csv(output_filename, index=False)

        print(
            f"\nОтримано {len(df_clean)} днів станційних вимірювань."
        )
        print(f"Файл збережено у '{output_filename}'")
        print("\nПерші 20 рядків:")
        print(df_clean.head(20))
        print(df_clean.describe())
    else:
        print("За даними датами нічого не знайдено.")

except Exception as e:
    print(f"Помилка завантаження: {e}")


#створюємо датафрейм для базового періоду
df_clean["dayofyear"] = df_clean["DATE"].dt.dayofyear
df_clean["year"] = df_clean["DATE"].dt.year
base_df = df_clean[(df_clean["year"]>=1991) & (df_clean["year"]<=2020)].copy()
print(base_df.tail(5))

#функція для обчисення 90-го перцентиля
def get_90pt_for_day(day, base_df):
    days_window = [(day + i -1) % 365 + 1 for i in range (-2,3)]

    subset = base_df[base_df["dayofyear"].isin(days_window)]
    return np.percentile(subset["Tmax"], 90)


unique_days=sorted(df_clean["dayofyear"].unique())
p90_dict = {d: get_90pt_for_day(d, base_df) for d in unique_days}
df_clean["T_p90"] = df_clean["dayofyear"].map(p90_dict)
df_clean["is_extreme_heat"] = df_clean["Tmax"] > df_clean["T_p90"]

print(df_clean.head(10))
print(df_clean["is_extreme_heat"].sum())

df_clean["hw_group_id"] = (df_clean["is_extreme_heat"] != df_clean["is_extreme_heat"].shift()).cumsum()
df_clean["block_len"] = df_clean.groupby("hw_group_id")["is_extreme_heat"].transform("count")
df_clean["is_heatwave"] = (df_clean["is_extreme_heat"] == True) & (df_clean["block_len"] >= 3)

""""
print("Днів у хвилях тепла:", df_clean["is_heatwave"].sum())
print(df_clean[df_clean["is_heatwave"]]["block_len"].max())

#літо 2005 року
summer_2005 = df_clean[(df_clean["year"] == 2005) & (df_clean["DATE"].dt.month.isin([6, 7, 8]))]

#літо 2025 року
summer_2025 = df_clean[(df_clean["year"] == 2025) & (df_clean["DATE"].dt.month.isin([6, 7, 8]))]

print(summer_2025[["DATE", "Tmax", "T_p90", "is_extreme_heat"]].head(60))
print(summer_2005[["DATE", "Tmax", "T_p90", "is_extreme_heat"]].head(60))

max_len = df_clean[df_clean["is_heatwave"]]["block_len"].max()
longest_heatwave = df_clean[(df_clean["is_heatwave"]) & (df_clean["block_len"] == max_len)]

print(f"Найдовша хвиля тепла тривала: {max_len} днів")
print(longest_heatwave[["DATE", "Tmax", "T_p90", "block_len"]])    """


#посушливі періоди
df_clean["is_dry_day"] = df_clean["PRCP"] < 1.0
df_clean["dry_group_id"] = (df_clean["is_dry_day"] != df_clean["is_dry_day"].shift()).cumsum()
df_clean["block_len_prcp"] = df_clean.groupby("dry_group_id")["is_dry_day"].transform("count")
df_clean["is_dry_spell"] = (df_clean["PRCP"] < 1.0) & (df_clean["block_len_prcp"] >= 5)

"""# Максимальна посуха за всі роки:
max_dry = df_clean[df_clean["is_dry_spell"]]["block_len_prcp"].max()
print(f"Найдовший посушливий період у Львові тривав: {max_dry} днів")

#коли була найдовша посуха:
print(df_clean[(df_clean["is_dry_spell"]) & (df_clean["block_len_prcp"] == max_dry)][["DATE", "PRCP", "block_len_prcp"]])
prcp_2008 = df_clean[(df_clean["year"] == 2008) & (df_clean["DATE"].dt.month.isin([5, 6, 7]))]
print(prcp_2008[["DATE", "PRCP"]].head(60))"""





#-------------------ДРУГИЙ ЕТАП!!!!!!!!!!-----------------------
yearly_df = pd.DataFrame()
#метрики спеки
heatwave_days = df_clean [df_clean["is_heatwave"]]

yearly_df["HWN"] = heatwave_days.groupby("year")["hw_group_id"].nunique()
yearly_df["HWD"] = heatwave_days.groupby("year")["block_len"].max()
yearly_df["HWF"] = df_clean.groupby("year")["is_heatwave"].sum()
yearly_df["HWA"] = heatwave_days.groupby("year")["Tmax"].max()

# Перший день року, коли зафіксовано хвилю
yearly_df["First_HW_Day"] = df_clean[df_clean["is_heatwave"]].groupby("year")["dayofyear"].min()
# Градуси перевищення порогу (тільки для днів у хвилях)
df_clean["heat_excess"] = (df_clean["Tmax"] - df_clean["T_p90"]).clip(lower=0)
# Сума накопиченого тепла за рік
yearly_df["HWC"] = df_clean[df_clean["is_heatwave"]].groupby("year")["heat_excess"].sum()
yearly_df["HWC"] = yearly_df["HWC"].fillna(0).round(1)
#тропічні ночі
df_clean["is_tropical_night"] = df_clean["Tmin"] >= 20.0
yearly_df["TR"] = df_clean.groupby("year")["is_tropical_night"].sum()


#метрики посухи
dry_spell_days = df_clean [df_clean["is_dry_spell"]]

yearly_df["DSN"] = dry_spell_days.groupby("year")["dry_group_id"].nunique()
yearly_df["DSD"] = dry_spell_days.groupby("year")["block_len_prcp"].max()
yearly_df["DSF"] = dry_spell_days.groupby("year")["is_dry_spell"].sum()

print(yearly_df)

"""yearly_df = yearly_df.reset_index()
yearly_df.to_excel('yearly_df_heat.xlsx', sheet_name='yearly', index=False)
"""



"""#-----------------------графік---------------------- (проміжний, не за кенделом. простий лінійний тренд)
sns.set_theme(style = "whitegrid")
plt.figure(figsize=(10,5))
plt.bar(yearly_df.index, yearly_df["HWF"], color = "#cc3d72", alpha=0.6, label= "Днів у хвилях спеки (HWF)")
plt.plot(yearly_df.index, yearly_df["HWD"], color = "darkred", marker="o", linewidth=2.5 , label= "Найдовша хвиля року (HWD, днів)")

years = yearly_df.index.values
hwf_values = yearly_df["HWF"].values
z = np.polyfit(years, hwf_values, 1)
p = np.poly1d(z)

plt.plot(years, p(years), color="#280f33", linestyle="--", linewidth=2, label=f"Лінія тренду спеки (нахил: +{z[0]:.2f} дн/рік)")
plt.title("Динаміка хвиль тепла у Львові 1990-2025", fontsize=16, fontweight="bold", pad=15)
plt.xlabel("Рік", fontsize=10)
plt.ylabel("Кількість днів", fontsize=10)
plt.xticks(years[::1], rotation=90)
plt.legend(fontsize=11, loc="upper left")

plt.tight_layout()
plt.show()
"""





# манн-кендалл тест
metrics = ["HWN", "HWD", "HWF", "HWA", "HWC", "TR", "DSN", "DSD", "DSF"]
results = []
for col in metrics:
    res = mk.original_test(yearly_df[col])
    results.append({
        "Метрика": col,
        "Тренд (Trend)": res.trend,  # 'increasing', 'decreasing' або 'no trend'
        "p-value": round(res.p, 4),  # Статистична значущість
        "Значущий (p < 0.05)": res.p < 0.05,  # True/False
        "Sen's Slope (Зміна/рік)": round(res.slope, 4),  # Швидкість зміни на рік
        "Tau (Кендалла)": round(res.Tau, 3)
    })

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
trend_summary = pd.DataFrame(results)
print(trend_summary)


df_clean["is_dry_spell_and_hw"] = (df_clean["is_dry_spell"]) & (df_clean["is_heatwave"])
total_hw_days = df_clean["is_heatwave"].sum()
compound_days = df_clean["is_dry_spell_and_hw"].sum()
compound_pct = (compound_days / total_hw_days) * 100

print(f"всього днів спеки: {total_hw_days}")
print(f"з них днів одночасно з посухою: {compound_days}")
print(f"частка хвиль спеки під час посухи: {compound_pct:.1f}%")

yearly_df["CD"] = df_clean.groupby("year")["is_dry_spell_and_hw"].sum()

# Обчислюємо кореляцію Спірмена в 10-річному рухомому вікні
yearly_df["Rolling_Corr"] = (
    yearly_df["HWF"]
    .rolling(window=10)
    .corr(yearly_df["DSF"])
)

print(yearly_df[["HWF", "DSF", "CD", "Rolling_Corr"]].tail(15))



#ГРАФІК ЗА 2024 РІК
# відбираємо дані за 2024 рік
df_2024 = df_clean[df_clean["year"] == 2024].copy()
# створюємо фігуру з 2 графіками (згори температура, знизу опади)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5), sharex=True, gridspec_kw={'height_ratios': [2.3, 1]})
sns.set_theme(style="whitegrid")

# верхнвй графік(температура)
ax1.plot(df_2024["DATE"], df_2024["Tmax"], color="black", linewidth=1.1, label="Фактична Tmax (°C)", alpha=0.8, zorder=3)
ax1.plot(df_2024["DATE"], df_2024["T_p90"], color="red", linestyle="--", linewidth=1.2, alpha=0.8, label="Кліматичний поріг (T_p90)", zorder=4)

# Виділяємо хвилі спеки червоним фоном
ax1.fill_between(
    df_2024["DATE"],
    df_2024["Tmax"],
    df_2024["T_p90"],
    where=df_2024["is_heatwave"],
    color="#ab0202",
    alpha=0.4,
    label="Хвиля спеки",
    zorder=5
)

ax1.set_title("Львів 2024 рік", fontsize=15, fontweight="bold", pad=12)
ax1.set_ylabel("Температура (°C)", fontsize=11)
ax1.legend(loc="upper left", frameon=True)
ax1.set_ylim(-15, 38)
ax1.axhline(0, color="#8d99ae", linestyle=":", linewidth=0.8, alpha=0.7)

# нижній графік (опади)
ax2.bar(df_2024["DATE"], df_2024["PRCP"], color="#577590", alpha=0.9, width=1, label="Опади (мм)", zorder=3)

# Виділяємо посушливі періоди (Dry Spells) жовтим фоном
ax2.fill_between(
    df_2024["DATE"],
    0,
    df_2024["PRCP"].max() + 5,
    where=df_2024["is_dry_spell"],
    color="#f9c74f",
    alpha=0.4,
    label="Посушливий період"
)

ax2.set_ylabel("Опади (мм)", fontsize=11)
ax2.set_xlabel("Місяць", fontsize=11, labelpad=8)
ax2.set_ylim(0, df_2024["PRCP"].max() + 3)
ax2.legend(loc="upper left", frameon=True)

for ax in (ax1, ax2):
  ax.grid(axis="y", color="#ebebeb", linestyle="-", linewidth=0.8, zorder=0)


# Форматування осі дат (показуємо місяці)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax2.xaxis.set_major_locator(mdates.MonthLocator())
ax2.set_xlim(df_2024["DATE"].min(), df_2024["DATE"].max())



plt.tight_layout()
plt.show()






sns.set_theme(style="whitegrid")

#два графіки (HWF та HWD поруч)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

years = yearly_df.index.values

# ГРАФІК 1: HWF (Загальна кількість днів спеки)
res_hwf = mk.original_test(yearly_df["HWF"])

# Точки даних та фактична лінія
ax1.plot(
    years,
    yearly_df["HWF"],
    marker="o",
    color="#6b317d",
    linewidth=1.7,
    label="Фактичні дані (HWF)",
)

# Розрахунок лінії Сена
# Точка перетину (intercept) обчислюється так, щоб лінія проходила через медіану даних
intercept_hwf = np.median(yearly_df["HWF"]) - res_hwf.slope * np.median(
    years - years[0]
)
sen_line_hwf = res_hwf.slope * (years - years[0]) + intercept_hwf

# Побудова лінії тренду Сена
ax1.plot(
    years,
    sen_line_hwf,
    color="darkred",
    linestyle="--",
    linewidth=2.5,
    label=f"Тренд Сена: +{res_hwf.slope:.3f} дн/рік\n(p-value = {res_hwf.p:.4f})",
)

ax1.set_title(
    "Динаміка днів у хвилях спеки (HWF)", fontsize=14, fontweight="bold", pad=10
)
ax1.set_xlabel("Рік", fontsize=11)
ax1.set_ylabel("Кількість днів", fontsize=11)
ax1.legend(loc="upper left", frameon=True, fontsize=10)


# ГРАФІК 2: HWD (Тривалість найдовшої хвилі за рік)
res_hwd = mk.original_test(yearly_df["HWD"])

ax2.plot(
    years,
    yearly_df["HWD"],
    marker="s",
    color="#c77a0e",
    linewidth=1.7,
    label="Фактичні дані (HWD)",
)

intercept_hwd = np.median(yearly_df["HWD"]) - res_hwd.slope * np.median(
    years - years[0]
)
sen_line_hwd = res_hwd.slope * (years - years[0]) + intercept_hwd

ax2.plot(
    years,
    sen_line_hwd,
    color="darkred",
    linestyle="--",
    linewidth=2.5,
    label=f"Тренд Сена: +{res_hwd.slope:.3f} дн/рік\n(p-value = {res_hwd.p:.4f})",
)

ax2.set_title(
    "Максимальна тривалість хвилі спеки (HWD)",
    fontsize=14,
    fontweight="bold",
    pad=10,
)
ax2.set_xlabel("Рік", fontsize=11)
ax2.set_ylabel("Тривалість (днів)", fontsize=11)
ax2.legend(loc="upper left", frameon=True, fontsize=10)

plt.suptitle(
    "Оцінка довгострокових трендів спеки у Львові (1990–2025)",
    fontsize=14,
    fontweight="bold",
    y=0.98,
)
plt.tight_layout()
plt.show()




"""# Налаштовуємо стиль
sns.set_theme(style="whitegrid")

# 1. Розраховуємо 10-річну скользячу кореляцію Спірмена між HWF та CD
window_size = 10
rolling_corr = (
    yearly_df["HWF"]
    .rolling(window=window_size)
    .corr(yearly_df["CD"])
)

# 2. Створюємо полотно для графіку
plt.figure(figsize=(12, 5))

years = yearly_df.index.values

# 3. Малюємо лінію скользячої кореляції
plt.plot(
    years,
    rolling_corr,
    color="#2b8a3e",
    linewidth=2.5,
    marker="o",
    label=f"{window_size}-річна скользяча кореляція (HWF vs CD)",
)

# 4. Додаємо горизонтальну лінію нульового зв'язку для орієнтиру
plt.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)

# 5. Зафарбовуємо область високої кореляції (> 0.7) для наочності
plt.fill_between(
    years,
    rolling_corr,
    0.7,
    where=(rolling_corr >= 0.7),
    color="#2b8a3e",
    alpha=0.2,
    label="Сильний зв'язок (r >= 0.7)",
)

# 6. Підписи та оформлення
plt.title(
    f"Динаміка зв'язку між спекою та посухою у Львові ({window_size}-річна Rolling Correlation)",
    fontsize=14,
    fontweight="bold",
    pad=12,
)
plt.xlabel("Рік (кінець 10-річного вікна)", fontsize=11)
plt.ylabel("Коефіцієнт кореляції Спірмена (r)", fontsize=11)
plt.ylim(-0.2, 1.05)
plt.yticks(np.arange(-0.2, 1.1, 0.2))
plt.legend(loc="lower right", frameon=True, fontsize=10)

plt.tight_layout()
plt.show()"""


rho, p_val = stats.spearmanr(yearly_df["HWF"], yearly_df["DSD"])

print(f"Загальна кореляція Спірмена (HWF vs DSD): {rho:.3f}")
print(f"p-value кореляції: {p_val:.4f} (Зв'язок значущий: {p_val < 0.05})")

# 2. 10-річна скользяча кореляція (Rolling Correlation)
yearly_df["Rolling_Corr_HWF_DSD"] = (
    yearly_df["HWF"]
    .rolling(window=10)
    .corr(yearly_df["DSD"])
)

print("\nОстанні значення Rolling Correlation (HWF vs DSD):")
print(yearly_df[["HWF", "DSD", "Rolling_Corr_HWF_DSD"]].tail(10))









#кількість днів посухи в теплий період
warm_season = df_clean[df_clean["DATE"].dt.month.isin([5, 6, 7, 8, 9])]
yearly_df["DSF_summer"] = warm_season.groupby("year")["is_dry_spell"].sum()
yearly_df["DSF_summer"] = yearly_df["DSF_summer"].fillna(0)

# 2. Розрахунок 10-річної скользячої кореляції Спірмена та p-value
window = 10
rolling_series = []
rolling_p_series = []

for i in range(len(yearly_df)):
  if i < window - 1:
    rolling_series.append(np.nan)
    rolling_p_series.append(np.nan)
  else:
    sub_df = yearly_df.iloc[i - window + 1 : i + 1]
    corr, p_val = stats.spearmanr(sub_df["HWF"], sub_df["DSF_summer"])
    rolling_series.append(corr)
    rolling_p_series.append(p_val)

yearly_df["Rolling_Spearman_Summer"] = rolling_series
yearly_df["Rolling_p_value"] = rolling_p_series

def check_significance(p):
  if pd.isna(p):
    return np.nan
  return "Так" if p < 0.05 else "Ні"

yearly_df["is_significant_corr"] = yearly_df["Rolling_p_value"].apply(
    check_significance
)

print(
    yearly_df[[
        "HWF",
        "DSF_summer",
        "Rolling_Spearman_Summer",
        "Rolling_p_value",
        "is_significant_corr",
    ]].tail(15)
)

# графік кореляції
fig, ax = plt.subplots(figsize=(11, 5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

years = yearly_df.index.values
r_values = yearly_df["Rolling_Spearman_Summer"].values
is_sig = yearly_df["is_significant_corr"] == "Так"

# Горизонтальні опорні лінії
ax.axhline(0, color="#b0b0b0", linestyle="-", linewidth=0.9, zorder=1)
ax.axhline(
    0.5,
    color="#d98099",
    linestyle=":",
    linewidth=1.3,
    alpha=0.9,
    label="Помірний зв'язок (r = 0.5)",
    zorder=1,
)

# Лінія скользячої кореляції
ax.plot(
    years,
    r_values,
    color="#1a102f",
    linewidth=2.2,
    label="Кореляція днів спеки та днів посухи у теплий період",
    zorder=2,
)

# точки, де p >= 0.05 (незначущий зв'язок)
valid_mask = ~np.isnan(r_values)
ax.scatter(
    years[~is_sig & valid_mask],
    r_values[~is_sig & valid_mask],
    color="#800000",
    edgecolor="#1a102f",
    s=35,
    zorder=3,
)

# точки, де p < 0.05 (значущий зв'язок)
if is_sig.sum() > 0:
  ax.scatter(
      years[is_sig],
      r_values[is_sig],
      color="#d98099",
      edgecolor="#800000",
      s=75,
      linewidth=1.5,
      zorder=4,
      label="Значущий зв'язок (p < 0.05)",
  )

# М'яка заливка зони позитивного зв'язку
ax.fill_between(
    years,
    r_values,
    0,
    where=(r_values > 0),
    color="#d98099",
    alpha=0.25,
    label="r > 0",
    zorder=1,
)

# Налаштування осей
ax.set_title(
    "Динаміка зв'язку між спекою та літньою посухою у Львові",
    fontsize=14,
    fontweight="bold",
    pad=15,
    color="#222222",
)
ax.set_ylabel("Коефіцієнт Спірмена ($r_s$)", fontsize=10.5, color="#333333")
ax.set_xlabel(
    "Рік (кінець 10-річного вікна)", fontsize=10.5, color="#333333", labelpad=8
)

ax.set_ylim(-0.6, 1.0)
#ax.set_xlim(years[0] - 0.5, years[-1] + 0.5)
ax.set_xticks(years[9:])
ax.set_xticklabels(years[9:], rotation=90, fontsize=9.5, color="#222222")

ax.grid(axis="y", color="#ebebeb", linestyle="-", linewidth=0.8, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#cccccc")
ax.spines["bottom"].set_color("#cccccc")

ax.legend(
    loc="lower left",
    frameon=True,
    facecolor="white",
    edgecolor="#e0e0e0",
    fontsize=9.5,
)

plt.tight_layout()
plt.show()


df_clean.to_csv("df_clean_prepared.csv", index=False)











sns.set_theme(style="whitegrid")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

years = yearly_df.index.values
x_rel = years - years[0]

# ГРАФІК 1: DSF
res_dsf = mk.original_test(yearly_df["DSF"])

ax1.plot(
    years,
    yearly_df["DSF"],
    marker="o",
    color="#0c2f6b",
    linewidth=1.7,
    label="Фактичні дані (DSF)",
)

# Класичний розрахунок інтерцепта Сена: median(Y - slope * X)
intercept_dsf = np.median(yearly_df["DSF"].values - res_dsf.slope * x_rel)
sen_line_dsf = res_dsf.slope * x_rel + intercept_dsf

ax1.plot(
    years,
    sen_line_dsf,
    color="#dbb21d",
    linestyle="--",
    linewidth=2.5,
    label=f"Тренд Сена: {res_dsf.slope:+.3f} дн/рік\n(p-value = {res_dsf.p:.4f})",
)

ax1.set_title(
    "Динаміка днів у періодах посухи (DSF)", fontsize=13, fontweight="bold", pad=10
)
ax1.set_xlabel("Рік", fontsize=11)
ax1.set_ylabel("Кількість днів", fontsize=11)
ax1.legend(loc="upper left", frameon=True, fontsize=10)



# ГРАФІК 2: DSD
res_dsd = mk.original_test(yearly_df["DSD"])

ax2.plot(
    years,
    yearly_df["DSD"],
    marker="s",
    color="#ba1488",
    linewidth=1.7,
    label="Фактичні дані (DSD)",
)

intercept_dsd = np.median(yearly_df["DSD"].values - res_dsd.slope * x_rel)
sen_line_dsd = res_dsd.slope * x_rel + intercept_dsd

ax2.plot(
    years,
    sen_line_dsd,
    color="#dbb21d",
    linestyle="--",
    linewidth=2.5,
    label=f"Тренд Сена: {res_dsd.slope:+.3f} дн/рік\n(p-value = {res_dsd.p:.4f})",
)

ax2.set_title(
    "Максимальна тривалість періоду посухи (DSD)",
    fontsize=13,
    fontweight="bold",
    pad=10,
)
ax2.set_xlabel("Рік", fontsize=11)
ax2.set_ylabel("Тривалість (днів)", fontsize=11)
ax2.legend(loc="upper left", frameon=True, fontsize=10)

plt.suptitle(
    "Тренди посухи у Львові (1990–2025)",
    fontsize=14,
    fontweight="bold",
    y=0.98,
)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.show()


yearly_df.to_csv("yearly_heatwaves.csv", index=False)