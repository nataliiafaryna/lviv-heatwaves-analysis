import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.dates as mdates


df = pd.read_csv("df_clean_prepared.csv", parse_dates=["DATE"])
df["DMT"] = (df["Tmax"] + df["Tmin"]) / 2.0

base_df = df[(df["year"] >= 1991) & (df["year"] <= 2020)]
dmt_95 = np.percentile(base_df["DMT"], 95)

df["DMT_3d"] = df["DMT"].rolling(window=3).mean()
df["DMT_30d"] = df["DMT"].shift(3).rolling(window=30).mean()

df["EHI_sig"] = df["DMT_3d"] - dmt_95
df["EHI_accl"] = df["DMT_3d"] - df["DMT_30d"]

df["EHF"] = df["EHI_sig"].clip(lower=0) * df["EHI_accl"].clip(lower=1)

base_ehf = df.loc[
    (df["year"] >= 1991) & (df["year"] <= 2020) & (df["EHF"] > 0), "EHF"
]
ehf_p85 = np.percentile(base_ehf, 85)
ehf_extreme_threshold = 3 * ehf_p85

print(f"Кліматичний поріг DMT_95: {dmt_95:.2f} °C")
print(f"Поріг Severe (EHF_p85): {ehf_p85:.2f} °C²")
print(f"Поріг Extreme (3 * EHF_p85): {ehf_extreme_threshold:.2f} °C²")
print(f"Історичний максимум EHF: {df['EHF'].max():.2f} °C²")


# 4. Категоризація
def categorize_ehf(ehf):
  if pd.isna(ehf) or ehf <= 0:
    return "None"
  elif ehf < ehf_p85:
    return "Moderate"
  elif ehf < ehf_extreme_threshold:
    return "Severe"
  else:
    return "Extreme"


df["EHF_category"] = df["EHF"].apply(categorize_ehf)

print("\nРозподіл днів за категоріями EHF (1990–2025):")
print(df["EHF_category"].value_counts())

#річні метрики
yearly_ehf = pd.DataFrame(index=sorted(df["year"].unique()))
yearly_ehf["EHF_days"] = (df[df["EHF"]>0].groupby("year")["EHF"].count().reindex(yearly_ehf.index, fill_value=0))
yearly_ehf["EHF_moderate_days"] = (
    df[df["EHF_category"] == "Moderate"]
    .groupby("year")["EHF"]
    .count()
    .reindex(yearly_ehf.index, fill_value=0)
)
yearly_ehf["EHF_severe_days"] = (
    df[df["EHF_category"] == "Severe"]
    .groupby("year")["EHF"]
    .count()
    .reindex(yearly_ehf.index, fill_value=0)
)


# Додаємо стовпець Extreme для повноти структури
yearly_ehf["EHF_extreme_days"] = (
    df[df["EHF_category"] == "Extreme"]
    .groupby("year")["EHF"]
    .count()
    .reindex(yearly_ehf.index, fill_value=0)
)


yearly_ehf["EHF_load"] = (
    df.groupby("year")["EHF"].sum().reindex(yearly_ehf.index, fill_value=0.0).round(1)
)
yearly_ehf["EHF_peak"] = (
    df.groupby("year")["EHF"].max().reindex(yearly_ehf.index, fill_value=0.0).round(2)
)

print(yearly_ehf.tail(10))

import pymannkendall as mk

metrics = ["EHF_days", "EHF_load", "EHF_peak"]
results = []

for m in metrics:
  res = mk.original_test(yearly_ehf[m])
  results.append({
      "Метрика": m,
      "Тренд": res.trend,
      "p-value": round(res.p, 4),
      "Значущий (p < 0.05)": res.p < 0.05,
      "Sen's Slope (Зміна/рік)": round(res.slope, 4),
      "Tau (Кендалла)": round(res.Tau, 3),
  })

ehf_trends_df = pd.DataFrame(results)
print(ehf_trends_df.to_string(index=False))



yearly_hw = pd.read_csv("yearly_heatwaves.csv")
hwf_values = yearly_hw["HWF"].values
# Формуємо порівняльний датафрейм
comparison_df = pd.DataFrame(
    {"HWF": hwf_values, "EHF_days": yearly_ehf["EHF_days"].values},
    index=yearly_ehf.index,
)

#Розрахунок кореляції Пірсона та Спірмена
corr_pearson, p_pearson = stats.pearsonr(
    comparison_df["HWF"], comparison_df["EHF_days"]
)
corr_spearman, p_spearman = stats.spearmanr(
    comparison_df["HWF"], comparison_df["EHF_days"]
)

print("кореляційний аналіз (HWF vs EHF_days):")
print(f"Кореляція Пірсона:  r = {corr_pearson:.3f} (p-value = {p_pearson:.4e})")
print(f"Кореляція Спірмена: rho = {corr_spearman:.3f} (p-value = {p_spearman:.4e})")

#Аналіз розбіжності (Дельта між методами)
comparison_df["Diff"] = comparison_df["HWF"] - comparison_df["EHF_days"]

print("\nстатистика розбіжності (HWF - EHF_days) :")
print(f"Сумарно днів HWF за 36 років: {comparison_df['HWF'].sum()}")
print(f"Сумарно днів EHF за 36 років: {comparison_df['EHF_days'].sum()}")
print(
    f"Загальна різниця: {comparison_df['HWF'].sum() - comparison_df['EHF_days'].sum()} днів"
)
print(f"Середня різниця на рік: {comparison_df['Diff'].mean():.1f} днів")
print(
    f"Максимальне перевищення HWF: {comparison_df['Diff'].max()} днів (у"
    f" {comparison_df['Diff'].idxmax()} році)"
)
print(
    f"Мінімальна різниця: {comparison_df['Diff'].min()} днів (у"
    f" {comparison_df['Diff'].idxmin()} році)"
)

# роки з найбільшою розбіжністю
print("\nТОП-5 років з найбільшою розбіжністю між методами:")
print(
    comparison_df.sort_values(by="Diff", ascending=False)[
        ["HWF", "EHF_days", "Diff"]
    ].head(5)
)




#візуалізація

#1 графік (порівняльний річний)
years = yearly_ehf.index.values
width = 0.38  # ширина стовпчиків

# Розрахунок лінії тренду Сена для EHF_load (нахил 3.377)
slope_load = 3.3770
intercept_load = np.median(
    yearly_ehf["EHF_load"].values - slope_load * (years - years[0])
)
trend_sen_load = intercept_load + slope_load * (years - years[0])

# 2. Побудова графіка
fig, ax1 = plt.subplots(figsize=(11, 5))
fig.patch.set_facecolor("white")
ax1.set_facecolor("white")

# Ліва вісь: згруповані стовпчики HWF та EHF_days
bars1 = ax1.bar(
    years - width / 2,
    comparison_df["HWF"],
    width=width,
    color="#ebcb17",
    alpha=0.85,
    label="Класичні дні спеки (HWF: Tmax > Tp90)",
)
bars2 = ax1.bar(
    years + width / 2,
    comparison_df["EHF_days"],
    width=width,
    color="#d65b27",
    alpha=0.9,
    label="Біокліматичні дні стресу (EHF > 0)",
)

ax1.set_ylabel("Кількість днів у році", fontsize=11, color="#222222")
ax1.set_xlabel("Рік", fontsize=11, color="#222222", labelpad=8)
ax1.set_xticks(years[::])
ax1.set_xticklabels(years[::], rotation=90, fontsize=9.5)
ax1.set_xlim(years[0] - 1, years[-1] + 1)
ax1.grid(axis="y", color="#e5e5e5", linestyle="--", linewidth=0.7, zorder=0)

# Права вісь: кумулятивне навантаження та тренд Сена
ax2 = ax1.twinx()
line1 = ax2.plot(
    years,
    yearly_ehf["EHF_load"],
    color="#206b32",
    marker="o",
    markersize=4,
    linewidth=1.8,
    label="EHFload (°C²)",
)
line2 = ax2.plot(
    years,
    trend_sen_load,
    color="#7a1710",
    linestyle="--",
    linewidth=2,
    label="Тренд Сена EHFload (+33.8 °C²/декада, p=0.002)",
)

ax2.set_ylabel(
    "Кумулятивне навантаження EHFload (°C²)",
    fontsize=11,
    color="#264653",
    labelpad=8,
)

lines_labels = [ax.get_legend_handles_labels() for ax in [ax1, ax2]]

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

all_handles = handles1 + handles2
all_labels = labels1 + labels2

ax1.legend(
    all_handles,
    all_labels,
    loc="upper left",
    ncol=2,
    frameon=True,
    facecolor="white",
    edgecolor="#cccccc",
    fontsize=8.5,
)
plt.title(
    "Порівняння метеорологічних (HWF) та біокліматичних (EHF) хвиль тепла у"
    " Львові (1990–2025)",
    fontsize=11,
    fontweight="bold",
    pad=15,
)

# Очищення рамок
for ax in (ax1, ax2):
  ax.spines["top"].set_visible(False)
  ax.spines["left"].set_color("#cccccc")
  ax.spines["bottom"].set_color("#cccccc")

plt.tight_layout()
plt.show()







#графік 2 (Stack Bar Chart суворості хвиль)
fig, ax = plt.subplots(figsize=(11, 5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

years = yearly_ehf.index.values

# 1. Стек-стовпчики (Moderate -> Severe -> Extreme)
b_mod = ax.bar(
    years,
    yearly_ehf["EHF_moderate_days"],
    color="#bd70cc",
    alpha=0.9,
    width=0.75,
    label="Помірний стрес (Moderate, EHF < 12.95 °C²)",
)

b_sev = ax.bar(
    years,
    yearly_ehf["EHF_severe_days"],
    bottom=yearly_ehf["EHF_moderate_days"],
    color="#800f7c",
    alpha=0.95,
    width=0.75,
    label="Суворий стрес (Severe, 12.95–38.85 °C²)",
)

b_ext = ax.bar(
    years,
    yearly_ehf["EHF_extreme_days"],
    bottom=yearly_ehf["EHF_moderate_days"] + yearly_ehf["EHF_severe_days"],
    color="#280ca6",
    alpha=0.95,
    width=0.75,
    label="Екстремальний стрес (Extreme, ≥ 38.85 °C²)",
)

# 2. Оформлення та сітка
ax.set_title(
    "Розподіл днів теплового стресу у Львові за категоріями суворості EHF"
    " (1990–2025)",
    fontsize=11.5,
    fontweight="bold",
    pad=15,
    color="#222222",
)
ax.set_ylabel("Кількість днів у році", fontsize=10.5, color="#222222")
ax.set_xlabel("Рік", fontsize=10.5, color="#222222", labelpad=8)

ax.set_xticks(years)
ax.set_xticklabels(years, rotation=90, fontsize=9, color="#222222")
ax.set_xlim(years[0] - 0.8, years[-1] + 0.8)

ax.grid(axis="y", color="#ebebeb", linestyle="--", linewidth=0.7, zorder=0)
ax.set_axisbelow(True)

# 3. Легенда та межі
ax.legend(
    loc="upper left",
    frameon=True,
    facecolor="white",
    edgecolor="#cccccc",
    fontsize=8.5,
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#cccccc")
ax.spines["bottom"].set_color("#cccccc")

plt.tight_layout()
plt.show()






#графік 3 (2024 рік)
df_2024 = df[
    (df["year"] == 2024) & (df["DATE"].dt.month.isin([5, 6, 7, 8, 9]))
].copy()

fig, ax = plt.subplots(figsize=(11, 5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# 2. Крива EHF та заливка зони теплового стресу
ax.plot(
    df_2024["DATE"],
    df_2024["EHF"],
    color="#6d4375",
    linewidth=2,
    label="EHF (°C²)",
    zorder=3,
)
ax.fill_between(
    df_2024["DATE"],
    0,
    df_2024["EHF"],
    where=(df_2024["EHF"] > 0),
    color="#6d4375",
    alpha=0.35,
    label="Зона теплового стресу (EHF > 0)",
    zorder=2,
)

# 3. Горизонтальні лінії порогів суворості
ax.axhline(
    y=ehf_p85,
    color="#b5179e",
    linestyle="--",
    linewidth=1.6,
    label=f"Поріг Severe ({ehf_p85:.2f} °C²)",
    zorder=4,
)
ax.axhline(
    y=ehf_extreme_threshold,
    color="#246dd4",
    linestyle=":",
    linewidth=1.6,
    label=f"Поріг Extreme ({ehf_extreme_threshold:.2f} °C²)",
    zorder=4,
)

# 4. Форматування осі дат (кожні 10-15 днів для читабельності)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
plt.xticks(rotation=90, fontsize=9, color="#222222")

# 5. Оформлення, підписи та сітка
ax.set_title(
    "Динаміка біокліматичного теплового стресу (EHF) у Львові: сезон 2024"
    " року",
    fontsize=12,
    fontweight="bold",
    pad=15,
    color="#222222",
)
ax.set_ylabel("Індекс EHF (°C²)", fontsize=10.5, color="#222222")
ax.set_xlabel("Дата", fontsize=10.5, color="#222222", labelpad=8)
ax.set_ylim(bottom=-1, top=max(df_2024["EHF"].max() + 5, 42))

ax.grid(axis="both", color="#ebebeb", linestyle="--", linewidth=0.7, zorder=1)
ax.set_axisbelow(True)

# 6. Легенда та межі
ax.legend(
    loc="upper left",
    frameon=True,
    facecolor="white",
    edgecolor="#cccccc",
    fontsize=9,
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#cccccc")
ax.spines["bottom"].set_color("#cccccc")

plt.tight_layout()
plt.show()