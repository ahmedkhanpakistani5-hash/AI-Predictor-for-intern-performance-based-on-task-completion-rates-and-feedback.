import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from groq import Groq


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Intern Performance AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# COMPLETE CSS THEME
# =========================================================

st.html("""
<style>

/* =====================================================
   MAIN APP BACKGROUND
   ===================================================== */

.stApp {
    background:
        radial-gradient(
            circle at 5% 10%,
            rgba(0, 210, 255, 0.15),
            transparent 25%
        ),
        radial-gradient(
            circle at 95% 5%,
            rgba(150, 45, 255, 0.20),
            transparent 28%
        ),
        radial-gradient(
            circle at 50% 100%,
            rgba(70, 40, 180, 0.12),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            #031329 0%,
            #061d3c 38%,
            #09184a 68%,
            #1b0b3d 100%
        );

    background-attachment: fixed;
}


/* =====================================================
   STREAMLIT TOP HEADER
   ===================================================== */

header[data-testid="stHeader"] {
    background:
        linear-gradient(
            90deg,
            #04152e 0%,
            #082b4d 35%,
            #101d55 65%,
            #35105b 100%
        ) !important;

    border-bottom:
        1px solid rgba(80, 220, 255, 0.35) !important;

    box-shadow:
        0 0 18px rgba(0, 200, 255, 0.25),
        0 0 35px rgba(110, 50, 255, 0.25) !important;
}


/* =====================================================
   NEON LINE UNDER TOP HEADER
   ===================================================== */

header[data-testid="stHeader"]::after {
    content: "";

    position: absolute;

    left: 0;
    right: 0;
    bottom: -2px;

    height: 3px;

    background:
        linear-gradient(
            90deg,
            #00eaff 0%,
            #008cff 30%,
            #634cff 55%,
            #a83cff 75%,
            #ef3cff 100%
        );

    box-shadow:
        0 0 8px #00eaff,
        0 0 18px #684cff,
        0 0 30px #c43cff;
}


/* =====================================================
   HEADER BUTTONS / ICONS
   ===================================================== */

header[data-testid="stHeader"] button {
    color: #dffaff !important;
}


/* =====================================================
   SIDEBAR
   ===================================================== */

section[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #041329 0%,
            #061a37 45%,
            #120b35 100%
        ) !important;

    border-right:
        1px solid rgba(0, 210, 255, 0.25);

    box-shadow:
        8px 0 35px rgba(0, 0, 0, 0.30);
}


section[data-testid="stSidebar"] * {
    color: #d9efff;
}


/* =====================================================
   MAIN CONTENT
   ===================================================== */

.block-container {
    max-width: 1400px !important;

    padding-top: 2.5rem !important;
    padding-bottom: 3rem !important;
}


/* =====================================================
   HERO
   ===================================================== */

.hero {
    position: relative;

    padding: 45px 42px;

    margin-bottom: 30px;

    border-radius: 26px;

    overflow: hidden;

    background:
        linear-gradient(
            135deg,
            rgba(4, 29, 57, 0.93),
            rgba(8, 29, 70, 0.91),
            rgba(39, 12, 70, 0.90)
        );

    border:
        1px solid rgba(0, 220, 255, 0.30);

    box-shadow:
        0 0 30px rgba(0, 190, 255, 0.12),
        0 0 55px rgba(130, 50, 255, 0.13);
}


.hero::before {
    content: "";

    position: absolute;

    width: 350px;
    height: 350px;

    top: -180px;
    left: -130px;

    background: #00d9ff;

    opacity: 0.10;

    filter: blur(80px);

    border-radius: 50%;
}


.hero::after {
    content: "";

    position: absolute;

    width: 400px;
    height: 400px;

    right: -150px;
    bottom: -200px;

    background: #a52cff;

    opacity: 0.12;

    filter: blur(90px);

    border-radius: 50%;
}


.hero-title {
    position: relative;

    z-index: 2;

    font-size: 3rem;

    font-weight: 850;

    background:
        linear-gradient(
            90deg,
            #ffffff,
            #58eaff,
            #8f8cff,
            #ff6ee7
        );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

    margin-bottom: 12px;
}


.hero-subtitle {
    position: relative;

    z-index: 2;

    color: #b9d9f3;

    font-size: 1.05rem;

    line-height: 1.7;

    max-width: 850px;
}


/* =====================================================
   SECTION TITLES
   ===================================================== */

.section-title {
    font-size: 1.45rem;

    font-weight: 800;

    color: #e0f7ff;

    margin-top: 30px;

    margin-bottom: 16px;

    text-shadow:
        0 0 15px rgba(0, 220, 255, 0.25);
}


/* =====================================================
   GLASS CARDS
   ===================================================== */

.glass-card {
    padding: 25px;

    border-radius: 20px;

    background:
        linear-gradient(
            135deg,
            rgba(7, 31, 61, 0.88),
            rgba(22, 19, 63, 0.84)
        );

    border:
        1px solid rgba(100, 210, 255, 0.18);

    box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.04),
        0 15px 45px rgba(0,0,0,0.20);

    backdrop-filter: blur(18px);

    margin-bottom: 20px;
}


/* =====================================================
   STAT CARDS
   ===================================================== */

.stat-card {
    padding: 22px;

    min-height: 125px;

    border-radius: 18px;

    background:
        linear-gradient(
            145deg,
            rgba(5, 31, 60, 0.92),
            rgba(27, 14, 61, 0.90)
        );

    border:
        1px solid rgba(85, 200, 255, 0.20);

    box-shadow:
        0 8px 35px rgba(0,0,0,0.22),
        0 0 20px rgba(80,80,255,0.06);

    transition: 0.25s ease;
}


.stat-card:hover {
    transform: translateY(-4px);

    border-color:
        rgba(0, 220, 255, 0.50);

    box-shadow:
        0 12px 40px rgba(0,0,0,0.25),
        0 0 25px rgba(0,180,255,0.18);
}


.stat-label {
    color: #91b8d9;

    font-size: 0.88rem;

    margin-bottom: 8px;
}


.stat-value {
    color: #ffffff;

    font-size: 1.65rem;

    font-weight: 800;
}


.stat-accent {
    color: #55e7ff;
}


/* =====================================================
   PREDICTION CARD
   ===================================================== */

.prediction-card {
    padding: 35px;

    margin-top: 20px;

    border-radius: 24px;

    background:
        linear-gradient(
            135deg,
            rgba(0, 105, 145, 0.30),
            rgba(72, 20, 125, 0.34)
        );

    border:
        1px solid rgba(0, 225, 255, 0.40);

    box-shadow:
        0 0 30px rgba(0, 200, 255, 0.12),
        0 0 50px rgba(140, 50, 255, 0.12);
}


.prediction-score {
    font-size: 3.5rem;

    font-weight: 900;

    background:
        linear-gradient(
            90deg,
            #53ecff,
            #8f8aff,
            #ff72dc
        );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}


.prediction-label {
    font-size: 1.25rem;

    color: #e8f8ff;

    font-weight: 750;
}


/* =====================================================
   BUTTON
   ===================================================== */

.stButton > button {
    width: 100%;

    border-radius: 14px !important;

    border:
        1px solid rgba(0, 220, 255, 0.45) !important;

    background:
        linear-gradient(
            90deg,
            #075c86,
            #3038a6,
            #702b9c
        ) !important;

    color: white !important;

    font-weight: 800 !important;

    padding: 0.75rem 1rem !important;

    box-shadow:
        0 0 18px rgba(0, 190, 255, 0.18);

    transition: 0.25s ease;
}


.stButton > button:hover {
    border-color: #70efff !important;

    box-shadow:
        0 0 25px rgba(0, 210, 255, 0.35),
        0 0 40px rgba(150, 60, 255, 0.20);

    transform: translateY(-2px);
}


/* =====================================================
   INPUT BOXES
   ===================================================== */

div[data-baseweb="input"] {
    background:
        rgba(5, 22, 48, 0.80) !important;

    border-radius: 12px !important;

    border:
        1px solid rgba(100, 190, 255, 0.22) !important;
}


div[data-baseweb="select"] > div {
    background:
        rgba(5, 22, 48, 0.80) !important;

    border-radius: 12px !important;
}


/* =====================================================
   SLIDER
   ===================================================== */

div[data-baseweb="slider"] {
    margin-bottom: 10px;
}


/* =====================================================
   EXPANDER
   ===================================================== */

div[data-testid="stExpander"] {
    background:
        rgba(7, 25, 55, 0.75) !important;

    border:
        1px solid rgba(100, 190, 255, 0.18) !important;

    border-radius: 16px !important;
}


/* =====================================================
   DATAFRAME
   ===================================================== */

[data-testid="stDataFrame"] {
    border-radius: 16px;

    overflow: hidden;

    border:
        1px solid rgba(0, 200, 255, 0.18);
}


/* =====================================================
   FOOTER
   ===================================================== */

.footer {
    margin-top: 50px;

    padding: 25px;

    text-align: center;

    color: #7293b4;

    border-top:
        1px solid rgba(100, 180, 255, 0.12);
}


/* =====================================================
   MOBILE
   ===================================================== */

@media (max-width: 768px) {

    .hero {
        padding: 30px 22px;
    }

    .hero-title {
        font-size: 2rem;
    }

    .prediction-score {
        font-size: 2.7rem;
    }

}

</style>
""")


# =========================================================
# CREATE INTERNAL DATASET
# =========================================================

@st.cache_data
def create_dataset():

    np.random.seed(42)

    n = 300

    task_completion_rate = np.random.uniform(
        45, 100, n
    )

    completion_time = np.random.uniform(
        1.5, 12, n
    )

    feedback_rating = np.random.uniform(
        1, 5, n
    )

    attendance = np.random.uniform(
        55, 100, n
    )

    performance = (
        task_completion_rate * 0.45
        + (12 - completion_time) * 3
        + feedback_rating * 7
        + attendance * 0.20
        + np.random.normal(0, 3, n)
    )

    performance = np.clip(
        performance,
        0,
        100
    )

    data = pd.DataFrame({
        "Task Completion Rate": task_completion_rate,
        "Task Completion Time": completion_time,
        "Feedback Rating": feedback_rating,
        "Attendance": attendance,
        "Performance Score": performance
    })

    return data


# =========================================================
# TRAIN RANDOM FOREST
# =========================================================

@st.cache_resource
def train_model():

    data = create_dataset()

    X = data[
        [
            "Task Completion Rate",
            "Task Completion Time",
            "Feedback Rating",
            "Attendance"
        ]
    ]

    y = data["Performance Score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=4,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    return (
        model,
        data,
        mae,
        rmse,
        r2,
        X_test,
        y_test,
        predictions
    )


model, data, mae, rmse, r2, X_test, y_test, predictions = train_model()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.html("""
    <div style="
        text-align:center;
        padding:10px 0 25px 0;
    ">

        <div style="
            font-size:2rem;
            margin-bottom:8px;
        ">
            📊
        </div>

        <div style="
            font-size:1.25rem;
            font-weight:800;
            color:#66eaff;
        ">
            Intern Performance AI
        </div>

        <div style="
            font-size:0.78rem;
            color:#789abd;
            margin-top:5px;
        ">
            Machine Learning Dashboard
        </div>

    </div>
    """)

    st.divider()

    st.markdown("### ⚙️ Model Information")

    st.write("**Algorithm:** Random Forest")
    st.write("**Task:** Regression")
    st.write("**Training Records:** 300")

    st.divider()

    st.markdown("### 🤖 AI Engine")

    st.write("Groq")
    st.write("GPT OSS 20B")

    st.divider()

    st.markdown("### 📈 Model Metrics")

    st.write(f"MAE: {mae:.2f}")
    st.write(f"RMSE: {rmse:.2f}")
    st.write(f"R² Score: {r2:.2f}")

    st.divider()

    st.caption(
        "Internal simulated dataset is used for demonstration."
    )


# =========================================================
# HERO
# =========================================================

st.html("""
<div class="hero">

    <div class="hero-title">
        Intern Performance AI
    </div>

    <div class="hero-subtitle">
        Predict intern performance using Machine Learning
        and generate AI-powered feedback using Groq.
        The system analyzes task completion rate,
        completion time, feedback rating, and attendance.
    </div>

</div>
""")


# =========================================================
# MODEL OVERVIEW
# =========================================================

st.html("""
<div class="section-title">
    📊 Model Overview
</div>
""")


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.html("""
    <div class="stat-card">

        <div class="stat-label">
            Model
        </div>

        <div class="stat-value">
            Random Forest
        </div>

    </div>
    """)


with col2:

    st.html("""
    <div class="stat-card">

        <div class="stat-label">
            Training Records
        </div>

        <div class="stat-value">
            <span class="stat-accent">
                300
            </span>
        </div>

    </div>
    """)


with col3:

    st.html(f"""
    <div class="stat-card">

        <div class="stat-label">
            R² Score
        </div>

        <div class="stat-value">
            <span class="stat-accent">
                {r2:.2f}
            </span>
        </div>

    </div>
    """)


with col4:

    st.html("""
    <div class="stat-card">

        <div class="stat-label">
            AI Engine
        </div>

        <div class="stat-value">
            Groq
        </div>

    </div>
    """)


# =========================================================
# INPUT SECTION
# =========================================================

st.html("""
<div class="section-title">
    🧑‍💻 Intern Performance Inputs
</div>
""")


left, right = st.columns(2)


with left:

    task_rate = st.slider(
        "Task Completion Rate (%)",
        min_value=0,
        max_value=100,
        value=80
    )

    completion_time = st.number_input(
        "Average Task Completion Time (hours)",
        min_value=1.0,
        max_value=20.0,
        value=5.0,
        step=0.5
    )


with right:

    feedback = st.slider(
        "Feedback Rating",
        min_value=1.0,
        max_value=5.0,
        value=4.0,
        step=0.1
    )

    attendance = st.slider(
        "Attendance (%)",
        min_value=0,
        max_value=100,
        value=90
    )


# =========================================================
# PREDICT BUTTON
# =========================================================

predict_button = st.button(
    "🚀 Predict Intern Performance",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    input_data = pd.DataFrame({

        "Task Completion Rate": [
            task_rate
        ],

        "Task Completion Time": [
            completion_time
        ],

        "Feedback Rating": [
            feedback
        ],

        "Attendance": [
            attendance
        ]

    })


    predicted_score = model.predict(
        input_data
    )[0]


    predicted_score = float(
        np.clip(
            predicted_score,
            0,
            100
        )
    )


    if predicted_score >= 75:

        category = "Likely to Excel"
        emoji = "🚀"

    elif predicted_score >= 55:

        category = "Average / Developing"
        emoji = "📈"

    else:

        category = "Needs Improvement"
        emoji = "⚠️"


    # =====================================================
    # RESULT
    # =====================================================

    st.html("""
    <div class="section-title">
        🎯 Prediction Result
    </div>
    """)


    st.html(f"""
    <div class="prediction-card">

        <div style="
            color:#8fb5d5;
            font-size:0.95rem;
            margin-bottom:10px;
        ">
            Predicted Performance Score
        </div>

        <div class="prediction-score">
            {predicted_score:.1f}/100
        </div>

        <div class="prediction-label">
            {emoji} {category}
        </div>

    </div>
    """)


    # =====================================================
    # INPUT SUMMARY
    # =====================================================

    st.html("""
    <div class="section-title">
        📋 Input Summary
    </div>
    """)


    s1, s2, s3, s4 = st.columns(4)


    with s1:

        st.metric(
            "Task Completion",
            f"{task_rate:.0f}%"
        )


    with s2:

        st.metric(
            "Completion Time",
            f"{completion_time:.1f} hrs"
        )


    with s3:

        st.metric(
            "Feedback",
            f"{feedback:.1f}/5"
        )


    with s4:

        st.metric(
            "Attendance",
            f"{attendance:.0f}%"
        )


    # =====================================================
    # GROQ AI
    # =====================================================

    st.html("""
    <div class="section-title">
        🤖 AI-Powered Performance Analysis
    </div>
    """)


    try:

        groq_key = st.secrets.get(
            "GROQ_API_KEY",
            ""
        )

    except Exception:

        groq_key = ""


    if groq_key:

        try:

            client = Groq(
                api_key=groq_key
            )


            prompt = f"""
You are an intern performance analyst.

Analyze this machine-learning prediction.

Task Completion Rate: {task_rate}%
Average Task Completion Time: {completion_time} hours
Feedback Rating: {feedback}/5
Attendance: {attendance}%
Predicted Performance Score: {predicted_score:.1f}/100
Prediction Category: {category}

Provide a concise professional analysis.

Use these sections:

### Performance Summary

### Key Strengths

### Areas to Improve

### Practical Recommendations

Give 3 practical recommendations.

Do not make assumptions about sensitive personal characteristics.
"""


            response = client.chat.completions.create(

                model="openai/gpt-oss-20b",

                messages=[

                    {
                        "role": "system",
                        "content":
                        "You are a professional intern performance analyst."
                    },

                    {
                        "role": "user",
                        "content": prompt
                    }

                ],

                temperature=0.4,

                max_tokens=700
            )


            ai_result = response.choices[0].message.content


            st.markdown(ai_result)


        except Exception as e:

            st.error(
                "Groq AI could not generate the analysis."
            )

            st.caption(
                f"Error: {str(e)}"
            )


    else:

        st.warning(
            "Groq API key is not configured."
        )

        st.info(
            "Add GROQ_API_KEY to Streamlit Secrets "
            "to enable AI analysis."
        )


    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    st.html("""
    <div class="section-title">
        🔍 Feature Importance
    </div>
    """)


    importance = pd.DataFrame({

        "Feature": [

            "Task Completion Rate",
            "Task Completion Time",
            "Feedback Rating",
            "Attendance"

        ],

        "Importance":
            model.feature_importances_

    }).sort_values(
        "Importance",
        ascending=False
    )


    st.bar_chart(
        importance.set_index("Feature")
    )


# =========================================================
# MODEL VALIDATION
# =========================================================

with st.expander(
    "📊 View Model Validation"
):

    validation_df = pd.DataFrame({

        "Actual Score":
            y_test.values,

        "Predicted Score":
            predictions

    })


    st.dataframe(
        validation_df.head(20),
        use_container_width=True
    )


    st.write(
        f"**Mean Absolute Error:** {mae:.2f}"
    )

    st.write(
        f"**Root Mean Squared Error:** {rmse:.2f}"
    )

    st.write(
        f"**R² Score:** {r2:.2f}"
    )


# =========================================================
# FOOTER
# =========================================================

st.html("""
<div class="footer">

    <div style="
        font-size:1rem;
        font-weight:700;
        color:#a9ddff;
        margin-bottom:7px;
    ">
        Intern Performance AI
    </div>

    <div>
        Machine Learning • Random Forest • Groq AI • Streamlit
    </div>

    <div style="
        margin-top:8px;
        font-size:0.75rem;
    ">
        Developed as part of an AI/ML internship project.
    </div>

</div>
""")
