import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
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
# GLOBAL CSS
# =========================================================

st.html("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 5% 15%,
            rgba(22, 111, 106, 0.45),
            transparent 28%
        ),
        radial-gradient(
            circle at 95% 20%,
            rgba(30, 66, 135, 0.48),
            transparent 30%
        ),
        radial-gradient(
            circle at 75% 90%,
            rgba(91, 49, 34, 0.42),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            #071016 0%,
            #0b1825 45%,
            #111a2d 72%,
            #1d110d 100%
        );
}


/* ================= SIDEBAR ================= */

section[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #11143b 0%,
            #15163d 50%,
            #0d1730 100%
        );

    border-right: 1px solid rgba(255,255,255,0.08);
}


/* ================= MAIN AREA ================= */

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}


/* ================= HERO ================= */

.hero {
    background:
        linear-gradient(
            135deg,
            rgba(7, 63, 76, 0.96),
            rgba(18, 37, 78, 0.97),
            rgba(55, 27, 67, 0.96)
        );

    border: 1px solid rgba(72, 211, 210, 0.18);

    border-radius: 22px;

    padding: 27px 30px;

    box-shadow:
        0 18px 55px rgba(0,0,0,0.38),
        0 0 35px rgba(26, 183, 188, 0.10);

    margin-bottom: 28px;
}

.hero-title {
    font-size: 38px;
    font-weight: 800;

    background:
        linear-gradient(
            90deg,
            #4ed9d2,
            #67dca5,
            #9b8cff
        );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

    margin-bottom: 8px;
}

.hero-subtitle {
    color: #e7ebf3;
    font-size: 15px;
    line-height: 1.65;
}


/* ================= SECTION TITLE ================= */

.section-title {
    font-size: 25px;
    font-weight: 800;

    color: #ffc96d;

    margin-top: 28px;
    margin-bottom: 18px;
}


/* ================= METRIC CARD ================= */

.metric-card {
    background:
        linear-gradient(
            145deg,
            rgba(48, 29, 24, 0.90),
            rgba(12, 34, 56, 0.94)
        );

    border:
        1px solid rgba(255,255,255,0.09);

    border-radius: 19px;

    padding: 20px;

    min-height: 115px;

    text-align: center;

    box-shadow:
        0 12px 35px rgba(0,0,0,0.30);
}

.metric-label {
    color: #91a4ba;
    font-size: 13px;
    margin-bottom: 9px;
}

.metric-value {
    color: #f4e7d8;
    font-size: 25px;
    font-weight: 800;
}


/* ================= INPUT CARD ================= */

.input-card {
    background:
        linear-gradient(
            145deg,
            rgba(27, 30, 62, 0.95),
            rgba(9, 43, 58, 0.94)
        );

    border:
        1px solid rgba(73, 209, 211, 0.13);

    border-radius: 18px;

    padding: 19px;

    margin-bottom: 10px;

    box-shadow:
        0 10px 30px rgba(0,0,0,0.28);
}

.input-icon {
    font-size: 27px;
}

.input-title {
    color: #f4f6fa;
    font-size: 16px;
    font-weight: 700;
    margin-top: 5px;
}

.input-description {
    color: #8fa4ba;
    font-size: 12px;
    margin-top: 4px;
}


/* ================= PREDICTION ================= */

.prediction-card {
    background:
        radial-gradient(
            circle at 20% 20%,
            rgba(41, 190, 183, 0.16),
            transparent 32%
        ),
        linear-gradient(
            135deg,
            rgba(10, 55, 65, 0.97),
            rgba(25, 31, 70, 0.97),
            rgba(53, 28, 27, 0.96)
        );

    border:
        1px solid rgba(75, 220, 211, 0.22);

    border-radius: 25px;

    padding: 35px;

    text-align: center;

    box-shadow:
        0 0 55px rgba(38, 194, 188, 0.12),
        0 20px 55px rgba(0,0,0,0.35);
}

.prediction-small {
    color: #a6b5c7;
    font-size: 16px;
}

.prediction-score {
    font-size: 65px;
    font-weight: 800;

    background:
        linear-gradient(
            90deg,
            #55dfd1,
            #72dfa5,
            #b29dff
        );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

    margin: 7px 0;
}

.prediction-status {
    color: #f4dfc8;
    font-size: 21px;
    font-weight: 700;
}


/* ================= AI CARD ================= */

.ai-card {
    background:
        linear-gradient(
            145deg,
            rgba(28, 24, 58, 0.96),
            rgba(8, 42, 57, 0.96)
        );

    border:
        1px solid rgba(136, 121, 255, 0.20);

    border-radius: 20px;

    padding: 26px;

    box-shadow:
        0 15px 45px rgba(0,0,0,0.32);
}


/* ================= BUTTON ================= */

.stButton > button {
    width: 100%;

    min-height: 52px;

    border-radius: 14px;

    border: 1px solid rgba(89, 218, 211, 0.25);

    background:
        linear-gradient(
            90deg,
            #164d58,
            #24396e,
            #4a2c49
        );

    color: white;

    font-size: 16px;
    font-weight: 700;

    box-shadow:
        0 9px 27px rgba(0,0,0,0.30);
}

.stButton > button:hover {
    border-color: #62ddd5;

    box-shadow:
        0 0 28px rgba(71,210,203,0.22);
}


/* ================= INPUTS ================= */

div[data-baseweb="input"] {
    background: rgba(7, 23, 38, 0.78);
    border-radius: 10px;
}

div[data-baseweb="input"] input {
    color: white !important;
}


/* ================= FOOTER ================= */

.footer {
    text-align: center;
    color: #7f91a5;
    font-size: 12px;
    padding: 30px 0;
}


/* ================= HIDE DEFAULT ================= */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

</style>
""")


# =========================================================
# DATA GENERATION
# =========================================================

@st.cache_data
def generate_dataset():

    np.random.seed(42)

    n = 300

    task_rate = np.random.uniform(45, 100, n)

    completion_time = np.random.uniform(1.5, 12, n)

    feedback = np.random.uniform(1, 5, n)

    attendance = np.random.uniform(55, 100, n)

    performance = (
        task_rate * 0.45
        + (12 - completion_time) * 3
        + feedback * 7
        + attendance * 0.20
        + np.random.normal(0, 3, n)
    )

    performance = np.clip(
        performance,
        0,
        100
    )

    return pd.DataFrame({
        "Task Completion Rate": task_rate,
        "Task Completion Time": completion_time,
        "Feedback Rating": feedback,
        "Attendance": attendance,
        "Performance Score": performance
    })


# =========================================================
# TRAIN MODEL
# =========================================================

@st.cache_resource
def train_model():

    data = generate_dataset()

    features = [
        "Task Completion Rate",
        "Task Completion Time",
        "Feedback Rating",
        "Attendance"
    ]

    X = data[features]
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

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

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
        features,
        mae,
        rmse,
        r2,
        y_test,
        predictions
    )


(
    model,
    data,
    features,
    mae,
    rmse,
    r2,
    y_test,
    predictions
) = train_model()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        "## ⚙️ Dashboard Settings"
    )

    st.markdown(
        """
        <div style="
            color:#8fc5ff;
            font-size:14px;
            line-height:1.8;
        ">

        <b>Model</b><br>
        Random Forest Regression

        <br><br>

        <b>AI Engine</b><br>
        Groq • GPT-OSS 20B

        <br><br>

        <b>Application</b><br>
        Streamlit

        <br><br>

        <b>Training Records</b><br>
        300 simulated interns

        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown(
        """
        <div style="
            color:#d5a6ff;
            font-size:16px;
            font-weight:700;
        ">
        📊 Model Metrics
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div style="
            color:#9bc9ff;
            line-height:2;
            font-size:14px;
        ">

        R² Score:
        <b>{r2:.2f}</b>

        <br>

        MAE:
        <b>{mae:.2f}</b>

        <br>

        RMSE:
        <b>{rmse:.2f}</b>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown(
        """
        <div style="
            color:#d5a6ff;
            font-size:15px;
            font-weight:700;
        ">
        🤖 AI Analysis
        </div>

        <div style="
            color:#9bc9ff;
            font-size:13px;
            line-height:1.6;
            margin-top:7px;
        ">
        AI analysis is securely connected
        through Streamlit Secrets.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="
            position:fixed;
            bottom:15px;
            left:18px;
            color:#d5bd70;
            font-size:13px;
        ">
        Build with Streamlit & AI
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# HERO
# =========================================================

st.html("""
<div class="hero">

    <div class="hero-title">
        📊 Intern Performance AI
    </div>

    <div class="hero-subtitle">
        Machine Learning powered performance prediction using
        task completion, feedback ratings and attendance —
        enhanced with Generative AI.
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


c1, c2, c3, c4 = st.columns(4)


with c1:

    st.html("""
    <div class="metric-card">

        <div class="metric-label">
            🤖 Model
        </div>

        <div class="metric-value">
            Random Forest
        </div>

    </div>
    """)


with c2:

    st.html(f"""
    <div class="metric-card">

        <div class="metric-label">
            🎯 R² Score
        </div>

        <div class="metric-value">
            {r2:.2f}
        </div>

    </div>
    """)


with c3:

    st.html(f"""
    <div class="metric-card">

        <div class="metric-label">
            📉 MAE
        </div>

        <div class="metric-value">
            {mae:.2f}
        </div>

    </div>
    """)


with c4:

    st.html(f"""
    <div class="metric-card">

        <div class="metric-label">
            👥 Training Records
        </div>

        <div class="metric-value">
            {len(data)}
        </div>

    </div>
    """)


# =========================================================
# INTERN ASSESSMENT
# =========================================================

st.html("""
<div class="section-title">
    👤 Intern Assessment
</div>
""")


left, right = st.columns(2)


# =========================================================
# LEFT SIDE
# =========================================================

with left:

    st.html("""
    <div class="input-card">

        <div class="input-icon">
            ✅
        </div>

        <div class="input-title">
            Task Completion Rate
        </div>

        <div class="input-description">
            Percentage of assigned tasks completed successfully.
        </div>

    </div>
    """)

    task_rate = st.slider(
        "Task Completion Rate (%)",
        0,
        100,
        82
    )


    st.html("""
    <div class="input-card">

        <div class="input-icon">
            ⏱️
        </div>

        <div class="input-title">
            Task Completion Time
        </div>

        <div class="input-description">
            Average time required to complete a task.
        </div>

    </div>
    """)

    completion_time = st.number_input(
        "Average Completion Time (hours)",
        min_value=0.5,
        max_value=50.0,
        value=5.0,
        step=0.5
    )


# =========================================================
# RIGHT SIDE
# =========================================================

with right:

    st.html("""
    <div class="input-card">

        <div class="input-icon">
            ⭐
        </div>

        <div class="input-title">
            Feedback Rating
        </div>

        <div class="input-description">
            Average feedback received from supervisors.
        </div>

    </div>
    """)

    feedback = st.slider(
        "Feedback Rating (1–5)",
        1.0,
        5.0,
        4.0,
        0.1
    )


    st.html("""
    <div class="input-card">

        <div class="input-icon">
            📅
        </div>

        <div class="input-title">
            Attendance
        </div>

        <div class="input-description">
            Percentage of working days attended.
        </div>

    </div>
    """)

    attendance = st.slider(
        "Attendance (%)",
        0,
        100,
        88
    )


# =========================================================
# PREDICT BUTTON
# =========================================================

st.write("")

predict = st.button(
    "🔮 Predict Intern Performance",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict:

    input_data = pd.DataFrame({
        "Task Completion Rate": [task_rate],
        "Task Completion Time": [completion_time],
        "Feedback Rating": [feedback],
        "Attendance": [attendance]
    })

    score = model.predict(input_data)[0]

    score = float(
        np.clip(
            score,
            0,
            100
        )
    )


    if score >= 75:

        status = "Likely to Excel"
        icon = "🚀"

    elif score >= 55:

        status = "Average / Developing"
        icon = "📈"

    else:

        status = "Needs Improvement"
        icon = "⚠️"


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

        <div class="prediction-small">
            Predicted Performance Score
        </div>

        <div class="prediction-score">
            {score:.1f}%
        </div>

        <div class="prediction-status">
            {icon} {status}
        </div>

        <div style="
            color:#8fa5b8;
            font-size:13px;
            margin-top:10px;
        ">
            Generated using Random Forest Regression
        </div>

    </div>
    """)


    # =====================================================
    # SUMMARY
    # =====================================================

    st.html("""
    <div class="section-title">
        📋 Assessment Summary
    </div>
    """)


    s1, s2, s3, s4 = st.columns(4)


    with s1:

        st.html(f"""
        <div class="metric-card">

            <div class="metric-label">
                Task Completion
            </div>

            <div class="metric-value">
                {task_rate}%
            </div>

        </div>
        """)


    with s2:

        st.html(f"""
        <div class="metric-card">

            <div class="metric-label">
                Completion Time
            </div>

            <div class="metric-value">
                {completion_time:.1f}h
            </div>

        </div>
        """)


    with s3:

        st.html(f"""
        <div class="metric-card">

            <div class="metric-label">
                Feedback
            </div>

            <div class="metric-value">
                {feedback:.1f}/5
            </div>

        </div>
        """)


    with s4:

        st.html(f"""
        <div class="metric-card">

            <div class="metric-label">
                Attendance
            </div>

            <div class="metric-value">
                {attendance}%
            </div>

        </div>
        """)


    # =====================================================
    # GROQ AI
    # =====================================================

    st.html("""
    <div class="section-title">
        🧠 Generative AI Analysis
    </div>
    """)


    # Get key from Streamlit Secrets ONLY
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
You are an expert internship performance analyst.

Analyze this machine-learning prediction.

Task Completion Rate: {task_rate}%
Average Task Completion Time: {completion_time} hours
Feedback Rating: {feedback}/5
Attendance: {attendance}%
Predicted Performance Score: {score:.1f}%
Performance Category: {status}

Give a concise professional analysis.

Use these sections:

### Performance Summary

### Key Strengths

### Areas to Improve

### Practical Recommendations

Focus only on the measurable performance data.
Do not make assumptions about personality,
health, background, or sensitive characteristics.
"""

            response = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a professional HR analytics "
                            "assistant."
                        )
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


            st.html("""
            <div class="ai-card">
            </div>
            """)

            st.markdown(
                ai_result
            )


        except Exception as error:

            st.error(
                "AI analysis could not be generated. "
                "Please check your Groq API configuration."
            )

    else:

        st.info(
            "AI analysis is not configured yet. "
            "Add GROQ_API_KEY to Streamlit Secrets."
        )


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

st.html("""
<div class="section-title">
    📊 Feature Importance
</div>
""")


importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=True
)


fig = plt.figure(
    figsize=(10, 4.5)
)

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.xlabel(
    "Relative Importance"
)

plt.title(
    "Random Forest Feature Importance"
)

plt.tight_layout()

st.pyplot(
    fig,
    use_container_width=True
)

plt.close(fig)


# =========================================================
# MODEL VALIDATION
# =========================================================

with st.expander("📈 View Model Validation"):

    comparison = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": predictions
    })

    fig2 = plt.figure(
        figsize=(9, 5)
    )

    plt.scatter(
        comparison["Actual"],
        comparison["Predicted"],
        alpha=0.7
    )

    plt.xlabel(
        "Actual Performance"
    )

    plt.ylabel(
        "Predicted Performance"
    )

    plt.title(
        "Actual vs Predicted Performance"
    )

    plt.tight_layout()

    st.pyplot(
        fig2,
        use_container_width=True
    )

    plt.close(fig2)


# =========================================================
# FOOTER
# =========================================================

st.html("""
<div class="footer">

    <b>Intern Performance AI</b>
    &nbsp; • &nbsp;
    Random Forest
    &nbsp; • &nbsp;
    Generative AI
    &nbsp; • &nbsp;
    Streamlit

    <br><br>

    Built with Machine Learning & AI

</div>
""")
