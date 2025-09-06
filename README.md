# پروژه: دیجیتال تویین هوشمند برای شبیه‌سازی خطوط تولید (Smart Digital Twin)

> **TL;DR**: این مخزن یک چارچوب پژوهشی-اجرایی برای ساخت یک **دیجیتال تویین** از خط تولید (2D/3D)، اتصال آن به **یادگیری عمیق** و **یادگیری تقویتی عمیق (DRL)**، و ارائه‌ی داشبورد تعاملی است. هدف، **بهینه‌سازی تولید** (کاهش زمان سیکل، WIP، نرخ خرابی) و **تصمیم‌گیری خودکار** (زمان‌بندی، تخصیص منابع) است.

---

## 1) تعریف دقیق مسئله (Problem Statement)

### 1.1 دامنه و سناریو
- **سناریو نمونه**: یک خط مونتاژ با دو یا سه ایستگاه کاری (Workstation)، یک ربات/اپراتور، و یک صف مشترک.
- **هدف شبیه‌سازی**: مدل‌سازی وضعیت‌های رخدادگسسته (ورود قطعه، سرویس، خرابی، تعمیر) و وضعیت‌های پیوسته (سرعت نقاله، دما).
- **دیجیتال تویین**: مدل دیجیتال که با داده‌های شبیه‌سازی یا واقعی همگام است و اجازه‌ی **آزمایش سناریو** و **کنترل بهینه** را می‌دهد.

### 1.2 اهداف تحقیق/پیاده‌سازی
- ساخت مدل دیجیتال تویین از خط تولید با امکان **3D Visualization** (Unity/AnyLogic) و **2D/داشبورد**.
- اتصال **شبکه‌های عصبی عمیق** برای پیش‌بینی گلوگاه‌ها (LSTM/Transformers).
- استفاده از **Deep RL** (DQN/PPO/A3C) برای **بهینه‌سازی تصمیم‌ها** (زمان‌بندی، تخصیص، سرعت خط).
- یکپارچه‌سازی **Simulation + AI** به صورت **real-time** و **آفلاین**.

### 1.3 داده‌ها و KPIها
- **ویژگی‌های داده**: زمان پردازش، خرابی/MTBF/MTTR، طول صف، موجودی WIP، کیفیت (عیوب)، مصرف انرژی.
- **KPIهای کلیدی**:
  - \(CT\): *Cycle Time*، زمان عبور.
  - \(TH\): *Throughput*، نرخ خروجی.
  - \(WIP\): موجودی در جریان ساخت.
  - \(OEE\): کارایی کلی تجهیزات.
  - \(Cost\): هزینه‌ی عملیاتی/انرژی.

### 1.4 قیود و مفروضات
- ظرفیت بافرها محدود، توزیع زمان پردازش (نمایی/لاگ-نرمال)، خرابی‌های تصادفی، محدودیت شیفت/اپراتور.
- **همگام‌سازی زمان** با شبیه‌ساز (گام‌زمان مجازی) و **دیلی شبکه** برای حلقه‌ی real-time.

### 1.5 تعریف مسئله‌ی بهینه‌سازی
- **RL Environment**: حالت‌ها (طول صف‌ها، مشغول/بیکار بودن ایستگاه‌ها، زمان‌های باقی‌مانده)، اقدامات (زمان‌بندی/ارسال به ایستگاه، تنظیم سرعت), پاداش (منفی CT/WIP/انرژی + مثبت TH/OEE).
- **اهداف**: کمینه‌سازی \(CT\) و انرژی، بیشینه‌سازی \(TH\) و \(OEE\)، رعایت قیود کیفیت.

---

## 2) معماری کلان سیستم (High-Level Architecture)

```
┌─────────────────────────┐       stream / batch       ┌──────────────────────────┐
│   2D/3D Simulator       │  ───────────────────────▶  │  Data Layer (Kafka/FS)   │
│   (SimPy / AnyLogic /   │◀────────────────────────── │  Parquet/CSV, Feature     │
│   Unity)                │   control / setpoints      │  Store (online/offline)   │
└──────────┬──────────────┘                            └──────────┬───────────────┘
           │  obs/reward/actions                                  │
           ▼                                                      ▼
┌─────────────────────────┐      training/inference     ┌──────────────────────────┐
│   RL/ML Engine          │◀───────────────────────────▶│  Orchestrator (Ray/Air)  │
│  (PyTorch/TensorFlow,   │                            │  + Experiment Manager     │
│   RLlib/Stable-Baselines)│                            └──────────┬───────────────┘
└──────────┬──────────────┘                                       │
           │                                                      ▼
           │                                      ┌──────────────────────────┐
           │                                      │  Visualization/Dashboard │
           └─────────────────────────────────────▶│  (Streamlit/Dash + 3D)   │
                                                  └──────────────────────────┘
```

### 2.1 اجزای اصلی
- **Simulator**: SimPy برای نمونه‌ی سبک + Unity/AnyLogic برای 3D.
- **Data Layer**: ذخیره‌ی جریان/بچ، تبدیل به Parquet/CSV، نگه‌داری **Feature Store** (آنلاین برای RL real-time، آفلاین برای آموزش).
- **RL/ML Engine**: آموزش DRL (DQN/PPO/A3C)، مدل‌های پیش‌بینی (LSTM/Transformer).
- **Orchestrator**: مدیریت آزمایش‌ها، تکرارها، لاگ‌ها (Ray/Airflow/Weights&Biases).
- **Visualization**: داشبورد KPI، حالت‌ها/اقدام‌ها، و رندر 3D.

### 2.2 حالت اجرای سیستم
- **Offline Loop**: شبیه‌سازی → جمع‌آوری → آموزش → ارزیابی → تکرار.
- **Online/Real-time Loop**: شبیه‌سازی/کارخانه → استنباط RL → اعمال فرمان → بازخورد.

### 2.3 یکپارچه‌سازی 3D
- **Unity**: پل ارتباطی با Python از طریق WebSocket/gRPC/ZeroMQ.
- **AnyLogic**: ارتباط از طریق Java API/Experiments + فایل/Socket/OPC UA.

---

## 3) روش‌ها و الگوریتم‌ها
- **DRL**:
  - *DQN*: مناسب فضای حالت گسسته؛ با شبکه‌ی dueling، prioritized replay.
  - *PPO*: پایدار، مناسب فضاهای پیوسته/بزرگ.
  - *A3C/A2C*: موازی‌سازی آسان، همگرایی سریع‌تر.
- **پیش‌بینی گلوگاه**: LSTM/Temporal Convolution/Transformers برای سری‌زمانی طول صف/زمان سیکل.
- **Hybrid**: قواعد تولید (Heuristics) + RL؛ warm-start با سیاست قواعدی.

---

## 4) داده‌ها و دیتاست‌ها
- **شبیه‌سازی‌شده**: تولید مصنوعی با NumPy/SimPy (زمان پردازش، خرابی، صف).
- **عمومی**: SECOM Manufacturing Data، Tennessee Eastman Process، Industrial IoT.
- **Schema پیشنهادی**:
  - `events.csv`: `timestamp, station_id, event_type(in|start|end|fail|repair), job_id`
  - `states.csv`: `timestamp, q_len_s1, q_len_s2, uptime_s1, uptime_s2, energy_kwh, defects`
  - `actions.csv`: `timestamp, policy, route(job, station), speed_setpoint`

---

## 5) ابزارها و تکنولوژی‌ها
- **زبان**: Python 3.10+
- **AI**: PyTorch یا TensorFlow؛ Stable-Baselines3/RLlib.
- **Simulation**: SimPy (پایه)، AnyLogic/Unity (3D).
- **Data/ETL**: Pandas, Polars, PyArrow; Kafka (اختیاری).
- **Viz/Dashboard**: Matplotlib/Plotly + Streamlit/Dash؛ رندر 3D (Unity/AnyLogic).
- **Integration**: WebSocket/gRPC/OPC UA.

---

## 6) ساختار مخزن (پیشنهادی)
```
DTwin-Factory/
├─ sims/                 # مدل‌های شبیه‌سازی (SimPy/Unity bridge)
│  ├─ miniline.py
│  ├─ unity_bridge.py
│  └─ anylogic_bridge.py
├─ rl/
│  ├─ envs/
│  │  └─ production_env.py
│  ├─ train_ppo.py
│  └─ train_dqn.py
├─ models/
│  └─ bottleneck_lstm.py
├─ data/
│  ├─ raw/
│  ├─ processed/
│  └─ features/
├─ dashboards/
│  └─ app_streamlit.py
├─ configs/
│  ├─ sim.yaml
│  └─ rl.yaml
├─ docs/
│  ├─ design.md
│  └─ experiments.md
├─ scripts/
│  ├─ generate_synthetic.py
│  └─ evaluate.py
├─ tests/
│  └─ test_env.py
├─ requirements.txt
└─ README.md
```

---

## 7) راه‌اندازی سریع (Setup)

### 7.1 پیش‌نیازها
- Python 3.10+
- (اختیاری) Unity 2022+ یا AnyLogic 8+

### 7.2 نصب
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

**requirements.txt (نمونه)**
```
simpy==4.1.1
numpy>=1.24
pandas>=2.0
pyyaml>=6.0
gymnasium>=0.29
torch>=2.1
streamlit>=1.31
plotly>=5.15
matplotlib>=3.8
mlflow>=2.12
orjson>=3.9
scikit-learn>=1.3
```

---

## 8) Quickstart (نمونه‌ی حداقلی)

### 8.1 شبیه‌ساز مینیمال (SimPy)
```python
# sims/miniline.py
import simpy, random

RANDOM_SEED = 42
PROC_MEAN = {"s1": 3.0, "s2": 4.0}  # میانگین زمان پردازش (دقیقه)
FAIL_PROB = {"s1": 0.02, "s2": 0.015}
REPAIR_TIME = {"s1": (5, 10), "s2": (6, 12)}  # یکنواخت

class Station:
    def __init__(self, env, name):
        self.env = env
        self.name = name
        self.machine = simpy.Resource(env, capacity=1)
        self.uptime = True

    def process(self, job_id):
        mean = PROC_MEAN[self.name]
        t = random.expovariate(1/mean)
        yield self.env.timeout(t)
        # خرابی تصادفی
        if random.random() < FAIL_PROB[self.name]:
            self.uptime = False
            low, high = REPAIR_TIME[self.name]
            yield self.env.timeout(random.uniform(low, high))
            self.uptime = True


def job_flow(env, s1: Station, s2: Station, metrics):
    jid = 0
    while True:
        jid += 1
        arrive = env.now
        with s1.machine.request() as req:
            yield req
            yield env.process(s1.process(jid))
        with s2.machine.request() as req:
            yield req
            yield env.process(s2.process(jid))
        ct = env.now - arrive
        metrics["ct"].append(ct)
        yield env.timeout(random.expovariate(1/3.5))  # ورود کارهای بعدی


def run(sim_time=480):
    random.seed(RANDOM_SEED)
    env = simpy.Environment()
    s1, s2 = Station(env, "s1"), Station(env, "s2")
    metrics = {"ct": []}
    env.process(job_flow(env, s1, s2, metrics))
    env.run(until=sim_time)
    return metrics

if __name__ == "__main__":
    m = run()
    print("AVG CT:", sum(m["ct"]) / len(m["ct"]))
```

### 8.2 اسکلت محیط RL (Gymnasium)
```python
# rl/envs/production_env.py
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class ProductionEnv(gym.Env):
    def __init__(self, sim_iface, cfg):
        super().__init__()
        self.sim = sim_iface  # wrapper برای SimPy/Unity/AnyLogic
        self.cfg = cfg
        self.observation_space = spaces.Box(low=0.0, high=1e3, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # مثال: ارسال به s1/s2 یا تنظیم سرعت

    def reset(self, *, seed=None, options=None):
        obs = self.sim.reset()
        return obs, {}

    def step(self, action):
        obs, kpis = self.sim.step(action)
        reward = self._reward(kpis)
        terminated = self.sim.done
        truncated = False
        info = {"kpis": kpis}
        return obs, reward, terminated, truncated, info

    def _reward(self, k):
        return -0.1*k["ct"] - 0.05*k["wip"] + 0.2*k["th"] - 0.01*k.get("energy", 0)
```

### 8.3 آموزش PPO (Stable-Baselines3)
```python
# rl/train_ppo.py
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from rl.envs.production_env import ProductionEnv
from sims.miniline import run  # یا یک wrapper مناسب

class SimWrapper:
    # این کلاس باید reset/step با obs و kpi برگرداند
    ...

if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/rl.yaml"))
    env = ProductionEnv(sim_iface=SimWrapper(), cfg=cfg)
    check_env(env, warn=True)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=200_000)
    model.save("artifacts/ppo_miniline")
```

---

## 9) ارزیابی و متریک‌ها
- **KPIs**: CT, TH, WIP, OEE, Energy/Cost، نرخ عیب.
- **Curves**: یادگیری (reward/episode)، histogram زمان‌های سیکل، طول صف.
- **Ablation**: بدون پیش‌بینی، با LSTM؛ بدون خرابی، با خرابی؛ Heuristic vs RL.
- **آزمون سناریو**: تغییر توزیع‌ها، ظرفیت بافر، سرعت نوار، سیاست تعمیرات.

---

## 10) داشبورد و نمایش (2D/3D)
- **Streamlit/Dash**: صفحه KPIها، کنترل‌های زنده، لاگ اپیزودها.
- **3D (Unity/AnyLogic)**: نمایش وضعیت ایستگاه‌ها، انیمیشن حرکت قطعات، رنگ‌بندی حالت‌ها.
- **اتصال زنده**: WebSocket/gRPC برای ارسال actions/observations در زمان واقعی.

**Streamlit (نمونه حداقلی)**
```python
# dashboards/app_streamlit.py
import streamlit as st
import pandas as pd
st.title("Smart Digital Twin Dashboard")
col1, col2, col3 = st.columns(3)
col1.metric("CT (avg)", "3.8 min")
col2.metric("TH (jobs/h)", "112")
col3.metric("WIP", "7")
st.line_chart(pd.DataFrame({"reward": [10,12,15,14,18,20]}))
```

---

## 11) راهنمای یکپارچه‌سازی 3D
- **Unity**
  - یک **Data Bridge** با Python (websocket/gRPC) پیاده‌سازی کنید.
  - Scene شامل نوار نقاله، ایستگاه‌ها، حسگرها؛ اسکریبتی که state/action را Serialize کند.
- **AnyLogic**
  - تجربه‌ی **Custom Experiment** با API جاوا؛ تبادل داده از فایل/سوکت/OPC UA.
- **سازگاری زمان**: نرخ به‌روزرسانی 10–30 Hz برای بصری‌سازی، 2–10 Hz برای RL real-time (بسته به پیچیدگی).

---

## 12) امنیت، حریم خصوصی، و اخلاق
- ناشناس‌سازی داده‌های واقعی؛ عدم انتشار هرگونه IP سازمانی.
- کنترل دسترسی به مدل و داده‌ها؛ ثبت رویدادها (audit).
- بررسی **ایمنی تصمیم‌ها** قبل از اعمال روی تجهیزات واقعی (HIL/SIL).

---

## 13) نقشه راه (Roadmap)
- [x] **مرحله 1: تعریف مسئله و معماری کلان** (این README)
- [ ] مرحله 2: ساخت شبیه‌ساز پایه (SimPy) و تولید داده مصنوعی
- [ ] مرحله 3: طراحی محیط Gym و آموزش PPO/DQN
- [ ] مرحله 4: پیش‌بینی گلوگاه با LSTM/Transformer
- [ ] مرحله 5: داشبورد Streamlit + نمودارهای KPI
- [ ] مرحله 6: اتصال 3D (Unity/AnyLogic) و حلقه‌ی real-time
- [ ] مرحله 7: ارزیابی جامع، گزارش علمی و بسته‌ی انتشار

---

## 14) نحوه اجرای آزمایش‌ها
```bash
# تولید داده مصنوعی
python scripts/generate_synthetic.py --out data/raw/events.csv

# اجرای شبیه‌ساز مینیمال
python sims/miniline.py

# آموزش PPO
python rl/train_ppo.py --config configs/rl.yaml

# اجرای داشبورد
streamlit run dashboards/app_streamlit.py
```

---

## 15) مستندسازی و گزارش
- `docs/design.md`: جزئیات معماری، تصمیم‌های طراحی.
- `docs/experiments.md`: تنظیمات آزمایش، نتایج، نمودارها، بحث.
- ساختار مقاله: مقدمه، پیشینه، روش، نتایج، بحث و نتیجه‌گیری.

---

## 16) لایسنس و ارجاع
- **License**: MIT 

```
@misc{smart_dt_2025,
  title  = {Smart Digital Twin for Manufacturing},
  author = {Your Name},
  year   = {2025},
  url    = {https://github.com/your/repo}
}
```

---

### نکته‌ی پایانی
اگر از Unity/AnyLogic استفاده می‌کنید، ابتدا حلقه‌ی **Offline** را کامل کنید (SimPy + RL + داشبورد)، سپس **Bridge 3D** را اضافه کنید. این کار ریسک را کاهش می‌دهد و دیباگ را ساده می‌کند.

