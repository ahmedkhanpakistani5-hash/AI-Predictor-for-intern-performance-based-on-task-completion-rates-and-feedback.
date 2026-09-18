import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from groq import Groq


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Intern Performance AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 10% 15%,
            rgba(44, 117, 108, 0.45),
            transparent 28%
        ),
        radial-gradient(
            circle at 88% 25%,
            rgba(27, 68, 130, 0.55),
            transparent 32%
        ),
        radial-gradient(
            circle at 70% 85%,
            rgba(92, 48, 34, 0.42),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            #071018 0%,
            #0b1724 42%,
            #111b2d 70%,
            #1b100d 100%
        );

    color: #f4f7fb;
}


/* =========================================================
   SIDEBAR
   ========================================================= */

section[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #11143b 0%,
            #15163e 45%,
            #10172f 100%
        );

    border-right: 1px solid rgba(255,255,255,0.08);
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.5rem;
}

.sidebar-title {
    font-size: 23px;
    font-weight: 800;
    color: #f5f7ff;
    margin-bottom: 18px;
}

.sidebar-section {
    color: #d4a8ff;
    font-size: 16px;
    font-weight: 700;
    margin-top: 25px;
    margin-bottom: 10px;
}

.sidebar-info {
    color: #9cc8ff;
    line-height: 1.8;
    font-size: 14px;
}

.sidebar-footer {
    position: fixed;
    bottom: 18px;
    left: 20px;
    color: #d6bd76;
    font-size: 13px;
}


/* =========================================================
   MAIN CONTAINER
   ========================================================= */

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}


/* =========================================================
   HERO
   ========================================================= */

.hero {
    background:
        linear-gradient(
            135deg,
            rgba(9, 50, 68, 0.95),
            rgba(18, 32, 73, 0.96),
            rgba(45, 23, 67, 0.94)
        );

    border: 1px solid rgba(77, 214, 214, 0.15);
    border-radius: 22px;

    padding: 25px 30px;

    box-shadow:
        0 15px 50px rgba(0,0,0,0.35),
        0 0 35px rgba(0, 174, 190, 0.12);

    margin-bottom: 25px;
}

.hero-title {
    font-size: 36px;
    font-weight: 800;

    background:
        linear-gradient(
            90deg,
            #45d5d0,
            #66d7a9,
            #a58aff
        );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

    margin-bottom: 8px;
}

.hero-subtitle {
    color: #e6e9f0;
    font-size: 15px;
    line-height: 1.6;
}


/* =========================================================
   SECTION TITLES
   ========================================================= */

.section-title {
    font-size: 24px;
    font-weight: 800;
    margin-top: 25px;
    margin-bottom: 17px;

    background:
        linear-gradient(
            90deg,
            #ffca70,
            #ffc35c,
            #70d7d2
        );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}


/* =========================================================
   GLASS CARDS
   ========================================================= */

.glass-card {
    background:
        linear-gradient(
            145deg,
            rgba(47, 29, 24, 0.82),
            rgba(11, 39, 61, 0.86)
        );

    border:
        1px solid rgba(255,255,255,0.09);

    border-radius: 20px;

    padding: 22px;

    box-shadow:
        0 12px 40px rgba(0,0,0,0.30),
        inset 0 1px 0 rgba(255,255,255,0.03);

    margin-bottom: 18px;
}


/* =========================================================
   INPUT CARDS
   ========================================================= */

.input-card {
    background:
        linear-gradient(
            145deg,
            rgba(28, 29, 63, 0.94),
            rgba(10, 40, 56, 0.92)
        );

    border: 1px solid rgba(79, 207, 211, 0.14);

    border-radius: 18px;

    padding: 18px;

    box-shadow:
        0 10px 30px rgba(0,0,0,0.28);
}

.input-icon {
    font-size: 26px;
}

.input-title {
    font-size: 15px;
    font-weight: 700;
    color: #f1f4fa;
    margin-top: 6px;
}

.input-description {
    font-size: 12px;
    color: #8fa5bd;
}


/* =========================================================
   METRIC CARDS
   ========================================================= */

.metric-card {
    background:
        linear-gradient(
            145deg,
            rgba(43, 27, 23, 0.92),
            rgba(13, 35, 58, 0.95)
        );

    border: 1px solid rgba(255,255,255,0.08);

    border-radius: 18px;

    padding: 18px;

    text-align: center;

    box-shadow:
        0 10px 35px rgba(0,0,0,0.25);
}

.metric-label {
    color: #91a3b8;
    font-size: 13px;
    margin-bottom: 7px;
}

.metric-value {
    color: #f3e5d7;
    font-size: 27px;
    font-weight: 800;
}


/* =========================================================
   PREDICTION RESULT
   ========================================================= */

.prediction-card {
    background:
        radial-gradient(
            circle at 20% 20%,
            rgba(30, 180, 171, 0.18),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            rgba(14, 53, 65, 0.95),
            rgba(26, 28, 65, 0.96),
            rgba(50, 27, 25, 0.94)
        );

    border:
        1px solid rgba(83, 220, 210, 0.22);

    border-radius: 25px;

    padding: 35px;

    text-align: center;

    box-shadow:
        0 0 50px rgba(26, 190, 187, 0.12),
        0 20px 50px rgba(0,0,0,0.35);
}

.prediction-score {
    font-size: 64px;
    font-weight: 800;

    background:
        linear-gradient(
            90deg,
            #55e0d0,
            #76dfa7,
            #b4a2ff
        );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.prediction-status {
    font-size: 21px;
    font-weight: 700;
    color: #f4dfca;
}


/* =========================================================
   AI CARD
   ========================================================= */

.ai-card {
    background:
        linear-gradient(
            145deg,
            rgba(26, 23, 57, 0.94),
            rgba(9, 42, 58, 0.95)
        );

    border: 1px solid rgba(143, 126, 255, 0.18);

    border-radius: 20px;

    padding: 25px;

    box-shadow:
        0 15px 45px rgba(0,0,0,0.30);
}


/* =========================================================
   BUTTON
   ========================================================= */

.stButton > button {
    width: 100%;

    min-height: 50px;

    border-radius: 13px;

    border: 1px solid rgba(91, 222, 215, 0.25);

    background:
        linear-gradient(
            90deg,
            #164b56,
            #24396d,
            #4b2d48
        );

    color: white;

    font-size: 16px;
    font-weight: 700;

    box-shadow:
        0 8px 25px rgba(0,0,0,0.30);

    transition: 0.2s ease;
}

.stButton > button:hover {
    border-color: #66ddd4;

    box-shadow:
        0 0 25px rgba(70,210,204,0.20);

    transform: translateY(-1px);
}


/* =========================================================
   SLIDERS
   ========================================================= */

div[data-testid="stSlider"] {
    padding-top: 4px;
}


/* =========================================================
   NUMBER INPUT
   ========================================================= */

div[data-baseweb="input"] {
    background: rgba(7, 22, 36, 0.75);
    border-radius: 10px;
}

div[data-baseweb="input"] input {
    color: #ffffff !important;
}


/* =========================================================
   SELECT BOX
   ========================================================= */

div[data-baseweb="select"] > div {
    background: rgba(7, 22, 36, 0.80);
    border-radius: 10px;
}


/* =========================================================
   FILE / DEFAULT STREAMLIT ELEMENTS
   ========================================================= */

[data-testid="stMetric"] {
    background: transparent;
}


/* =========================================================
   HIDE STREAMLIT BRANDING
   ========================================================= */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}


/* =========================================================
   FOOTER
   ========================================================= */

.custom-footer {
    text-align: center;
    color: #8392a5;
    font-size: 12px;
    padding-top: 30px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# GENERATE INTERNAL DATASET
# =========================================================

@st.cache_data
def generate_dataset():

    np.random.seed(42)

    number_of_interns = 300

    task_completion_rate = np.random.uniform(
        45, 100, number_of_interns
    )

    task_completion_time = np.random.uniform(
        1.5, 12, number_of_interns
    )

    feedback_rating = np.random.uniform(
        1, 5, number_of_interns
    )

    attendance = np.random.uniform(
        55, 100, number_of_interns
    )

    performance = (
        task_completion_rate * 0.45
        + (12 - task_completion_time) * 3.0
        + feedback_rating * 7.0
        + attendance * 0.20
        + np.random.normal(0, 3, number_of_interns)
    )

    performance = np.clip(
        performance,
        0,
        100
    )

    dataset = pd.DataFrame({
        "Task Completion Rate": task_completion_rate,
        "Task Completion Time": task_completion_time,
        "Feedback Rating": feedback_rating,
        "Attendance": attendance,
        "Performance Score": performance
    })

    return dataset


# =========================================================
# TRAIN RANDOM FOREST
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

    model.fit(
        X_train,
        y_train
    )

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
        '<div class="sidebar-title">⚙️ Dashboard Settings</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="sidebar-info">

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
        {len(data)} simulated interns

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-section">🔑 API Configuration</div>',
        unsafe_allow_html=True
    )

    groq_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="Enter your Groq API key",
        label_visibility="collapsed"
    )

    st.caption(
        "Used only during the current session."
    )

    st.divider()

    st.markdown(
        """
        <div class="sidebar-info">

        <b>Model Metrics</b><br><br>

        R² Score:
        """,
        unsafe_allow_html=True
    )

    st.write(f"**{r2:.2f}**")

    st.markdown(
        f"""
        <div class="sidebar-info">

        MAE:
        <b>{mae:.2f}</b><br><br>

        RMSE:
        <b>{rmse:.2f}</b>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-footer">Build with Streamlit & AI</div>',
        unsafe_allow_html=True
    )


# =========================================================
# HERO
# =========================================================

st.markdown(
    """
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
    """,
    unsafe_allow_html=True
)


# =========================================================
# MODEL OVERVIEW
# =========================================================

st.markdown(
    '<div class="section-title">📈 Model Overview</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.markdown(
        f"""
        <div class="metric-card">

            <div class="metric-label">
                🤖 Model
            </div>

            <div class="metric-value">
                Random Forest
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

with c2:

    st.markdown(
        f"""
        <div class="metric-card">

            <div class="metric-label">
                🎯 R² Score
            </div>

            <div class="metric-value">
                {r2:.2f}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

with c3:

    st.markdown(
        f"""
        <div class="metric-card">

            <div class="metric-label">
                📉 MAE
            </div>

            <div class="metric-value">
                {mae:.2f}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

with c4:

    st.markdown(
        f"""
        <div class="metric-card">

            <div class="metric-label">
                👥 Training Records
            </div>

            <div class="metric-value">
                {len(data)}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# INTERN ASSESSMENT
# =========================================================

st.markdown(
    '<div class="section-title">👤 Intern Assessment</div>',
    unsafe_allow_html=True
)

left, right = st.columns(2)


# ---------------------------------------------------------
# LEFT INPUTS
# ---------------------------------------------------------

with left:

    st.markdown(
        """
        <div class="input-card">

            <div class="input-icon">✅</div>

            <div class="input-title">
                Task Completion Rate
            </div>

            <div class="input-description">
                Percentage of assigned tasks completed successfully
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    task_rate = st.slider(
        "Task Completion Rate (%)",
        0,
        100,
        82,
        key="task_rate"
    )

    st.markdown(
        """
        <div class="input-card">

            <div class="input-icon">⏱️</div>

            <div class="input-title">
                Task Completion Time
            </div>

            <div class="input-description">
                Average time required to complete an assigned task
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    completion_time = st.number_input(
        "Average Completion Time (hours)",
        min_value=0.5,
        max_value=50.0,
        value=5.0,
        step=0.5,
        key="completion_time"
    )


# ---------------------------------------------------------
# RIGHT INPUTS
# ---------------------------------------------------------

with right:

    st.markdown(
        """
        <div class="input-card">

            <div class="input-icon">⭐</div>

            <div class="input-title">
                Feedback Rating
            </div>

            <div class="input-description">
                Average feedback received from supervisors
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    feedback = st.slider(
        "Feedback Rating (1–5)",
        1.0,
        5.0,
        4.0,
        0.1,
        key="feedback"
    )

    st.markdown(
        """
        <div class="input-card">

            <div class="input-icon">📅</div>

            <div class="input-title">
                Attendance
            </div>

            <div class="input-description">
                Percentage of working days attended
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    attendance = st.slider(
        "Attendance (%)",
        0,
        100,
        88,
        key="attendance"
    )


# =========================================================
# PREDICT BUTTON
# =========================================================

st.markdown("<br>", unsafe_allow_html=True)

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

        status = "Likely to Excel"
        icon = "🚀"

    elif predicted_score >= 55:

        status = "Average / Developing"
        icon = "📈"

    else:

        status = "Needs Improvement"
        icon = "⚠️"


    # =====================================================
    # RESULT
    # =====================================================

    st.markdown(
        '<div class="section-title">🎯 Prediction Result</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="prediction-card">

            <div style="
                font-size:18px;
                color:#a9b8ca;
                margin-bottom:5px;
            ">
                Predicted Performance Score
            </div>

            <div class="prediction-score">
                {predicted_score:.1f}%
            </div>

            <div class="prediction-status">
                {icon} {status}
            </div>

            <div style="
                margin-top:12px;
                color:#91a5b9;
                font-size:13px;
            ">
                Generated using Random Forest Regression
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # =====================================================
    # INPUT SUMMARY
    # =====================================================

    st.markdown(
        '<div class="section-title">📋 Assessment Summary</div>',
        unsafe_allow_html=True
    )

    s1, s2, s3, s4 = st.columns(4)

    with s1:
        st.markdown(
            f"""
            <div class="metric-card">

            <div class="metric-label">
            Task Completion
            </div>

            <div class="metric-value">
            {task_rate}%
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with s2:
        st.markdown(
            f"""
            <div class="metric-card">

            <div class="metric-label">
            Completion Time
            </div>

            <div class="metric-value">
            {completion_time:.1f}h
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with s3:
        st.markdown(
            f"""
            <div class="metric-card">

            <div class="metric-label">
            Feedback
            </div>

            <div class="metric-value">
            {feedback:.1f}/5
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with s4:
        st.markdown(
            f"""
            <div class="metric-card">

            <div class="metric-label">
            Attendance
            </div>

            <div class="metric-value">
            {attendance}%
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # =====================================================
    # GROQ AI ANALYSIS
    # =====================================================

    st.markdown(
        '<div class="section-title">🧠 Generative AI Analysis</div>',
        unsafe_allow_html=True
    )

    if groq_key:

        try:

            client = Groq(
                api_key=groq_key
            )

            prompt = f"""
You are an expert internship performance analyst.

Analyze the following machine-learning prediction.

Task Completion Rate: {task_rate}%
Average Task Completion Time: {completion_time} hours
Feedback Rating: {feedback}/5
Attendance: {attendance}%
Predicted Performance Score: {predicted_score:.1f}%
Category: {status}

Provide a professional and concise analysis.

Use exactly these sections:

### Performance Summary
Explain the overall prediction.

### Key Strengths
Mention the strongest measurable factors.

### Areas to Improve
Mention measurable areas that could improve.

### Practical Recommendations
Give 2 or 3 actionable recommendations.

Do not make assumptions about personality, health,
background, or other sensitive personal characteristics.
Focus only on the provided performance data.
"""

            response = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a professional HR analytics "
                            "assistant focused on objective "
                            "performance analysis."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.4,
                max_tokens=800
            )

            ai_result = response.choices[0].message.content

            st.markdown(
                f"""
                <div class="ai-card">
                """,
                unsafe_allow_html=True
            )

            st.markdown(ai_result)

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )

        except Exception as error:

            st.error(
                "Groq AI analysis could not be generated. "
                "Please verify your API key."
            )

    else:

        st.info(
            "🔑 Enter your Groq API key in the sidebar to "
            "generate the AI-powered analysis."
        )


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

st.markdown(
    '<div class="section-title">📊 What Influences Performance?</div>',
    unsafe_allow_html=True
)

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=True
)

fig = plt.figure(figsize=(10, 4.5))

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.xlabel("Relative Importance")
plt.title("Random Forest Feature Importance")

plt.tight_layout()

st.pyplot(
    fig,
    use_container_width=True
)

plt.close(fig)


# =========================================================
# ACTUAL VS PREDICTED
# =========================================================

with st.expander("📈 View Model Validation"):

    comparison = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": predictions
    })

    fig2 = plt.figure(figsize=(9, 5))

    plt.scatter(
        comparison["Actual"],
        comparison["Predicted"],
        alpha=0.7
    )

    plt.xlabel("Actual Performance")
    plt.ylabel("Predicted Performance")

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

st.markdown(
    """
    <div class="custom-footer">

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
    """,
    unsafe_allow_html=True
)
