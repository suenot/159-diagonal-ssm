# Глава 138: Диагональные модели пространства состояний для трейдинга

В этой главе рассматриваются **диагональные модели пространства состояний (Diagonal SSMs)** — упрощённый, но высокоэффективный вариант структурированных моделей пространства состояний (S4), который достигает сопоставимой производительности при значительно сниженной вычислительной сложности. Ограничивая матрицу состояний **A** диагональной формой, такие модели обеспечивают эффективное параллельное обучение и быстрый инференс, что делает их идеальными для прогнозирования финансовых временных рядов.

## Содержание

1. [Введение в диагональные SSM](#введение-в-диагональные-ssm)
    * [От S4 к диагональным SSM](#от-s4-к-диагональным-ssm)
    * [Ключевые преимущества](#ключевые-преимущества)
    * [Сравнение с другими вариантами SSM](#сравнение-с-другими-вариантами-ssm)
2. [Математические основы](#математические-основы)
    * [Основы моделей пространства состояний](#основы-моделей-пространства-состояний)
    * [Диагональная параметризация](#диагональная-параметризация)
    * [Дискретизация](#дискретизация)
    * [Эффективная свёртка](#эффективная-свёртка)
3. [Практические примеры](#практические-примеры)
    * [01: Подготовка данных](#01-подготовка-данных)
    * [02: Архитектура диагональной SSM](#02-архитектура-диагональной-ssm)
    * [03: Обучение модели](#03-обучение-модели)
    * [04: Прогнозирование финансовых временных рядов](#04-прогнозирование-финансовых-временных-рядов)
    * [05: Бэктестирование стратегии](#05-бэктестирование-стратегии)
4. [Реализация на Rust](#реализация-на-rust)
5. [Реализация на Python](#реализация-на-python)
6. [Лучшие практики](#лучшие-практики)
7. [Ресурсы](#ресурсы)

## Введение в диагональные SSM

### От S4 к диагональным SSM

Модель структурированных пространств состояний для последовательностей (S4) ввела инициализированную с помощью HiPPO матрицу состояний для захвата дальних зависимостей в последовательностях. Однако S4 требует сложных методов матричной декомпозиции (например, нормальная плюс низкоранговая, или NPLR) для работы с плотной матрицей состояний **A**, что приводит к сложной реализации и процедурам обучения.

**Диагональные пространства состояний** (DSS), представленные Gupta и др. (2022) в статье "Diagonal State Spaces are as Effective as Structured State Spaces", продемонстрировали, что простое ограничение матрицы состояний диагональной формой достигает производительности наравне с S4, при этом радикально упрощая как реализацию, так и вычисления.

```
Архитектура S4 (сложная):
┌─────────────────────────────────────────────────┐
│  x(t) = A·x(t-1) + B·u(t)                     │
│  y(t) = C·x(t) + D·u(t)                        │
│                                                  │
│  A ∈ ℂ^{N×N}  (Dense, requires NPLR decomp.)   │
│  Eigenvalue decomposition needed                 │
│  Complex Cauchy kernel computation               │
└─────────────────────────────────────────────────┘

Диагональная SSM (простая):
┌─────────────────────────────────────────────────┐
│  x(t) = Λ·x(t-1) + B·u(t)                     │
│  y(t) = C·x(t) + D·u(t)                        │
│                                                  │
│  Λ = diag(λ₁, λ₂, ..., λₙ) ∈ ℂ^N             │
│  Each state evolves independently!               │
│  Trivially parallelizable                        │
└─────────────────────────────────────────────────┘
```

### Ключевые преимущества

1. **Упрощённая реализация**
   - Не требуется NPLR-декомпозиция
   - Диагональная структура тривиально диагонализируема
   - Стандартное автоматическое дифференцирование работает без дополнительных настроек

2. **Вычислительная эффективность**
   - O(N) на обновление состояния вместо O(N²)
   - Тривиальная параллелизация по измерениям состояния
   - Эффективная свёртка через БПФ (FFT)

3. **Сопоставимая производительность**
   - Соответствует S4 на бенчмарках Long Range Arena
   - Высокие результаты на последовательном CIFAR, речи, тексте
   - Эффективна для финансовых временных рядов

4. **Стабильность обучения**
   - Более простой ландшафт оптимизации
   - Более простые стратегии инициализации
   - Более устойчивый поток градиентов

### Сравнение с другими вариантами SSM

| Модель | Матрица состояний | Сложность | Параллелизуемость | Реализация |
|--------|------------------|-----------|--------------------|------------|
| **Diagonal SSM** | Диагональная Λ | O(NL) | Да (тривиально) | Простая |
| S4 | NPLR A | O(NL log L) | Да (сложно) | Сложная |
| S4D | Диагональная (вариант S4) | O(NL) | Да | Средняя |
| Mamba | Зависимая от данных | O(NL) | Да (scan) | Средняя |
| Linear RNN | Плотная A | O(N²L) | Нет | Простая |
| HiPPO-RNN | HiPPO A | O(N²L) | Нет | Средняя |

## Математические основы

### Основы моделей пространства состояний

Модель пространства состояний непрерывного времени отображает входной сигнал u(t) в выходной сигнал y(t) через скрытое состояние x(t):

```
dx/dt = A·x(t) + B·u(t)
  y(t) = C·x(t) + D·u(t)
```

Где:
- **A** ∈ ℝ^{N×N}: Матрица перехода состояний
- **B** ∈ ℝ^{N×1}: Проекция входа
- **C** ∈ ℝ^{1×N}: Проекция выхода
- **D** ∈ ℝ: Прямое соединение (сквозная связь)
- **N**: Размерность состояния (размер скрытого пространства)

### Диагональная параметризация

В диагональных SSM мы ограничиваем матрицу A диагональной формой:

```
A = Λ = diag(λ₁, λ₂, ..., λₙ)
```

Это означает, что каждое измерение состояния эволюционирует независимо:

```
dxᵢ/dt = λᵢ·xᵢ(t) + Bᵢ·u(t)    for i = 1, ..., N
```

Собственные значения λᵢ являются комплекснозначными, что позволяет моделировать колебательное поведение:

```
λᵢ = aᵢ + j·bᵢ    where aᵢ < 0 (stability constraint)
```

**Инициализация**: Следуя DSS, диагональные элементы можно инициализировать следующими способами:
1. **На основе HiPPO**: Аппроксимация собственных значений матрицы HiPPO
2. **Лог-равномерная**: λᵢ = -exp(uniform(log(0.001), log(0.1))) + j·uniform(0, π)
3. **S4D-Lin**: λₙ = -1/2 + j·π·n для равномерного распределения

### Дискретизация

Для применения непрерывной модели к дискретным последовательностям с шагом Δ мы выполняем дискретизацию методом фиксации нулевого порядка (ZOH):

```
Ā = exp(Λ·Δ)           (element-wise for diagonal)
B̄ = (Ā - I) · Λ⁻¹ · B  (simplified for diagonal)

Discrete recurrence:
x[k] = Ā·x[k-1] + B̄·u[k]
y[k] = C·x[k] + D·u[k]
```

Для диагональной Λ матричная экспонента вычисляется поэлементно:

```
Ā = diag(exp(λ₁·Δ), exp(λ₂·Δ), ..., exp(λₙ·Δ))
```

### Эффективная свёртка

SSM можно вычислить как свёртку с ядром K:

```
K = (C·B̄, C·Ā·B̄, C·Ā²·B̄, ..., C·Ā^{L-1}·B̄)

y = K * u    (convolution, computed via FFT in O(L log L))
```

Для диагональной Ā каждый элемент K равен:

```
K[l] = Σᵢ Cᵢ · Āᵢˡ · B̄ᵢ    for l = 0, ..., L-1
```

Это можно эффективно вычислить с помощью произведения Вандермонда:

```
K = C ⊙ B̄ · V(Ā, L)

where V(Ā, L) = [1, Ā, Ā², ..., Ā^{L-1}]  (element-wise powers)
```

## Практические примеры

### 01: Подготовка данных

Мы используем как данные фондового рынка (через Yahoo Finance), так и данные криптовалют (через API Bybit).

```python
from python.data import BybitClient, StockDataLoader, DiagonalSSMDataset

# Fetch crypto data from Bybit
client = BybitClient()
btc_klines = client.get_klines("BTCUSDT", interval="60", limit=1000)
eth_klines = client.get_klines("ETHUSDT", interval="60", limit=1000)

# Fetch stock data
stock_loader = StockDataLoader()
spy_data = stock_loader.get_stock_data("SPY", period="2y")

# Create dataset with features
dataset = DiagonalSSMDataset(
    data=btc_klines,
    seq_len=168,        # 1 week of hourly data
    pred_horizon=24,    # Predict 24 hours ahead
    features=["returns", "volatility", "volume_change", "rsi", "macd"]
)
```

### 02: Архитектура диагональной SSM

```python
from python.model import DiagonalSSMConfig, DiagonalSSMForecaster

config = DiagonalSSMConfig(
    input_features=6,
    state_dim=64,
    d_model=128,
    num_layers=4,
    dropout=0.1,
    init_method="s4d_lin",   # S4D-Lin initialization
    bidirectional=False,
    prediction_horizon=24
)

model = DiagonalSSMForecaster(config)
```

### 03: Обучение модели

```python
import torch
from python.model import DiagonalSSMForecaster, DiagonalSSMConfig

config = DiagonalSSMConfig(state_dim=64, d_model=128, num_layers=4)
model = DiagonalSSMForecaster(config)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

for epoch in range(100):
    for batch in train_loader:
        x, y = batch
        pred = model(x)
        loss = torch.nn.functional.mse_loss(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
    scheduler.step()
```

### 04: Прогнозирование финансовых временных рядов

Диагональная SSM обрабатывает финансовые временные ряды, трактуя каждый признак (цена, объём, индикаторы) как входной канал и пропуская его через несколько диагональных SSM-слоёв, которые захватывают временные зависимости на разных масштабах.

```
Input Features          Diagonal SSM Layers          Prediction
┌──────────┐    ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐
│ Returns   │───>│        │─>│        │─>│        │─>│        │
│ Volume    │───>│ Layer  │─>│ Layer  │─>│ Layer  │─>│ Return │
│ RSI       │───>│   1    │─>│   2    │─>│   3    │─>│ Pred.  │
│ MACD      │───>│        │─>│        │─>│        │─>│        │
│ Volatility│───>│        │─>│        │─>│        │─>│        │
└──────────┘    └────────┘  └────────┘  └────────┘  └────────┘
                 (State=64)  (State=64)  (State=64)
                 Diagonal Λ  Diagonal Λ  Diagonal Λ
```

### 05: Бэктестирование стратегии

```python
from python.strategy import BacktestConfig, Backtester

config = BacktestConfig(
    initial_capital=100000.0,
    commission=0.001,
    slippage=0.0005,
    long_threshold=0.001,
    short_threshold=-0.001,
    stop_loss_level=0.05
)

backtester = Backtester(config)
result = backtester.run(model, test_dataset)

print(f"Total Return: {result.total_return:.2%}")
print(f"Sharpe Ratio: {result.sharpe_ratio:.3f}")
print(f"Sortino Ratio: {result.sortino_ratio:.3f}")
print(f"Max Drawdown: {result.max_drawdown:.2%}")
```

## Реализация на Rust

Реализация на Rust предоставляет высокопроизводительную версию модели Diagonal SSM с интеграцией API Bybit:

```rust
use diagonal_ssm::{BybitClient, DataLoader, DiagonalSSMModel, DiagonalSSMConfig};

#[tokio::main]
async fn main() {
    // Fetch data from Bybit
    let client = BybitClient::new();
    let klines = client.get_klines("BTCUSDT", "60", 1000).await.unwrap();

    // Prepare dataset
    let loader = DataLoader::new()
        .seq_len(168)
        .target_horizon(24);
    let dataset = loader.prepare_dataset(&klines).unwrap();

    // Create and train model
    let config = DiagonalSSMConfig {
        state_dim: 64,
        d_model: 128,
        num_layers: 4,
        ..Default::default()
    };
    let mut model = DiagonalSSMModel::new(config);
    model.train(&dataset, 100, 1e-3).unwrap();

    // Run backtest
    let results = model.backtest(&dataset).unwrap();
    println!("Sharpe: {:.3}", results.sharpe_ratio);
}
```

Полную реализацию смотрите в директории `rust/`.

## Реализация на Python

Реализация на Python использует PyTorch для модели и предоставляет:
- `python/model.py`: Диагональный SSM-слой и модель прогнозирования
- `python/data.py`: Загрузка данных из Bybit и Yahoo Finance
- `python/strategy.py`: Фреймворк бэктестирования с управлением рисками
- `python/example_usage.py`: Полный пример использования

Полный исходный код смотрите в директории `python/`.

## Лучшие практики

1. **Инициализация имеет значение**: Используйте инициализацию S4D-Lin или на основе HiPPO для диагональных элементов. Случайная инициализация часто приводит к плохой сходимости.

2. **Комплексные vs вещественные числа**: Использование комплекснозначных диагональных элементов позволяет модели захватывать колебательные паттерны (характерные для финансовых рынков). Убедитесь, что ваша реализация корректно обрабатывает комплексную арифметику.

3. **Шаг дискретизации Δ**: Шаг дискретизации является обучаемым параметром для каждого слоя. Инициализируйте его лог-равномерным распределением в диапазоне [0.001, 0.1].

4. **Ограничение устойчивости**: Сохраняйте вещественные части диагональных собственных значений отрицательными (Re(λᵢ) < 0) для обеспечения устойчивости. Применяйте репараметризацию через softplus или экспоненту.

5. **Нормализация**: Применяйте слоевую нормализацию (Layer Normalization) после каждого SSM-слоя. Это стабилизирует обучение и улучшает обобщающую способность.

6. **Финансовые данные**: Нормализуйте доходности и признаки к нулевому среднему и единичной дисперсии. Используйте скользящие статистики для избежания ошибки заглядывания вперёд (lookahead bias).

7. **Длина последовательности**: Диагональные SSM эффективно обрабатывают длинные последовательности. Используйте более длинные окна наблюдения (168-720 часов) для лучшего захвата паттернов.

## Ресурсы

1. **Diagonal State Spaces are as Effective as Structured State Spaces** — Gupta, Hasani, Sontag (2022)
   - URL: https://arxiv.org/abs/2203.14343

2. **Efficiently Modeling Long Sequences with Structured State Spaces (S4)** — Gu, Goel, Ré (2022)
   - URL: https://arxiv.org/abs/2111.00396

3. **On the Parameterization and Initialization of Diagonal State Space Models (S4D)** — Gu, Gupta, Goel, Ré (2022)
   - URL: https://arxiv.org/abs/2206.11893

4. **How to Train Your HiPPO** — Gu, Johnson, Timeseries, Dao, Rudra, Ré (2022)
   - URL: https://arxiv.org/abs/2206.12037
