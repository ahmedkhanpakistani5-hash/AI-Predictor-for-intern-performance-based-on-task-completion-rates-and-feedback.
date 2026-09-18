import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from groq import Groq
import os

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="Intern Performance AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(67, 42, 30, 0.35), transparent 35%),
        radial-gradient(circle at 90% 20%, rgba(20, 42, 70, 0.45), transparent 35%),
        linear-gradient(135deg, #090d14 0%, #111827 48%, #1b120e 100%);
    color: #f5f1eb;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

/* Main title */

.hero {
    padding: 30px;
    border-radius: 24px;
    background: linear-gradient(
        135deg,
        rgba(42, 27, 21, 0.94),
        rgba(12, 29, 52, 0.94)
    );
    border: 1px solid rgba(190, 160, 130, 0.18);
    box-shadow: 0 20px 60px rgba(0,0,0,0.35);
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 8px;
    color: #f5e9dc;
}

.hero p {
    font-size: 17px;
    color: #b8c2cf;
}

/* Cards */

.card {
    background: linear-gradient(
        145deg,
        rgba(48, 30, 22, 0.78),
        rgba(14, 31, 53, 0.82)
    );
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 22px;
    box-shadow: 0 15px 40px rgba(0,0,0,0.25);
    margin-bottom: 20px;
}

.metric-card {
    background: linear-gradient(
        135deg,
        rgba(53, 33, 23, 0.95),
        rgba(14, 35, 60, 0.95)
    );
    border: 1px solid rgba(214, 184, 155, 0.15);
    border-radius: 18px;
    padding: 20px;
    text-align: center;
    min-height: 125px;
}

.metric-title {
    color: #9eabbc;
    font-size: 14px;
    margin-bottom: 8px;
}

.metric-value {
    color: #f3e8dc;
    font-size: 30px;
    font-weight: 700;
}

/* Section headers */

.section-title {
    font-size: 25px;
    font-weight: 700;
    color: #ead8c8;
    margin-top: 15px;
    margin-bottom: 15px;
}

/* Buttons */

.stButton > button {
    width: 100%;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.12);
    background: linear-gradient(135deg, #4a3025, #173452);
    color: white;
    font-weight: 600;
    padding: 12px;
}

.stButton > button:hover {
    border-color: #b89a82;
    color: white;
}

/* Inputs */

div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    background-color: rgba(13, 25, 40, 0.85);
    border-radius: 10px;
}

.stTextInput input,
.stNumberInput input {
    color: white !important;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #120d0a,
        #0d1b2d
    );
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* Dataframe */

div[data-testid="stDataFrame"] {
    border-radius: 15px;
    overflow: hidden;
}

/* Hide default menu */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.markdown("""
<div class="hero">

<h1>📊 Intern Performance AI</h1>

<p>
Machine Learning powered performance prediction using task completion,
feedback ratings and attendance — enhanced with Generative AI.
</p>

</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.markdown("## ⚙️ Dashboard Settings")

st.sidebar.markdown("""
**Model:** Random Forest Regression  
**AI:** Groq  
**Application:** Streamlit  
**Purpose:** Intern Performance Prediction
""")

st.sidebar.divider()

groq_key = st.sidebar.text_input(
    "🔑 Groq API Key",
    type="password",
    placeholder="Enter your Groq API key"
)

st.sidebar.caption(
    "Your API key is used only for the current session."
)


# ---------------------------------------------------------
# SAMPLE DATA
# ---------------------------------------------------------

@st.cache_data
def create_sample_data():

    np.random.seed(42)

    n = 150

    task_completion_rate = np.random.uniform(45, 100, n)

    task_completion_time = np.random.uniform(1, 12, n)

    feedback_rating = np.random.uniform(1, 5, n)

    attendance = np.random.uniform(55, 100, n)

    performance = (
        task_completion_rate * 0.45
        + (12 - task_completion_time) * 3
        + feedback_rating * 7
        + attendance * 0.20
        + np.random.normal(0, 3, n)
    )

    performance = np.clip(performance, 0, 100)

    df = pd.DataFrame({
        "Task Completion Rate (%)": task_completion_rate.round(2),
        "Task Completion Time (hours)": task_completion_time.round(2),
        "Feedback Rating": feedback_rating.round(2),
        "Attendance (%)": attendance.round(2),
        "Performance Score": performance.round(2)
    })

    return df


# ---------------------------------------------------------
# DATA SECTION
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">📁 Training Data</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns([1, 1])

with col1:

    st.markdown("""
    <div class="card">
    <b>Use Sample Dataset</b><br>
    Train the model immediately using generated intern data.
    </div>
    """, unsafe_allow_html=True)

    use_sample = st.button(
        "🚀 Load Sample Dataset",
        use_container_width=True
    )

with col2:

    uploaded_file = st.file_uploader(
        "Upload your CSV dataset",
        type=["csv"]
    )


if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)
    st.success("CSV dataset loaded successfully.")

elif use_sample or "data" not in st.session_state:

    data = create_sample_data()
    st.session_state["data"] = data

else:

    data = st.session_state["data"]


# ---------------------------------------------------------
# DATA PREVIEW
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">🔎 Dataset Preview</div>',
    unsafe_allow_html=True
)

st.dataframe(
    data.head(10),
    use_container_width=True,
    hide_index=True
)


# ---------------------------------------------------------
# CHECK REQUIRED COLUMNS
# ---------------------------------------------------------

required_columns = [
    "Task Completion Rate (%)",
    "Task Completion Time (hours)",
    "Feedback Rating",
    "Attendance (%)",
    "Performance Score"
]

missing_columns = [
    col for col in required_columns
    if col not in data.columns
]

if missing_columns:

    st.error(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )

    st.stop()


# ---------------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------------

features = [
    "Task Completion Rate (%)",
    "Task Completion Time (hours)",
    "Feedback Rating",
    "Attendance (%)"
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


# ---------------------------------------------------------
# MODEL EVALUATION
# ---------------------------------------------------------

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(y_test, predictions)


st.markdown(
    '<div class="section-title">🤖 Model Performance</div>',
    unsafe_allow_html=True
)

m1, m2, m3 = st.columns(3)

with m1:
    st.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-title">R² Score</div>
        <div class="metric-value">{r2:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with m2:
    st.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-title">MAE</div>
        <div class="metric-value">{mae:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with m3:
    st.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-title">RMSE</div>
        <div class="metric-value">{rmse:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------------
# FEATURE IMPORTANCE
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">📈 Feature Importance</div>',
    unsafe_allow_html=True
)

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=True
)

fig = plt.figure(figsize=(9, 4))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.xlabel("Importance")
plt.title("What Influences Intern Performance?")

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)


# ---------------------------------------------------------
# PREDICTION SECTION
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">🎯 Predict Intern Performance</div>',
    unsafe_allow_html=True
)

p1, p2 = st.columns(2)

with p1:

    task_rate = st.slider(
        "Task Completion Rate (%)",
        min_value=0,
        max_value=100,
        value=80
    )

    completion_time = st.number_input(
        "Average Task Completion Time (hours)",
        min_value=0.1,
        max_value=50.0,
        value=5.0,
        step=0.5
    )

with p2:

    feedback = st.slider(
        "Feedback Rating",
        min_value=1.0,
        max_value=5.0,
        value=4.0,
        step=0.1
    )

    attendance_value = st.slider(
        "Attendance (%)",
        min_value=0,
        max_value=100,
        value=85
    )


predict_button = st.button(
    "🔮 Predict Intern Performance",
    use_container_width=True
)


if predict_button:

    input_data = pd.DataFrame({
        "Task Completion Rate (%)": [task_rate],
        "Task Completion Time (hours)": [completion_time],
        "Feedback Rating": [feedback],
        "Attendance (%)": [attendance_value]
    })

    predicted_score = model.predict(input_data)[0]

    predicted_score = np.clip(
        predicted_score,
        0,
        100
    )

    # Classification based on predicted score
    if predicted_score >= 75:
        status = "Likely to Excel"
        emoji = "🚀"
    elif predicted_score >= 55:
        status = "Average / Developing"
        emoji = "📈"
    else:
        status = "Needs Improvement"
        emoji = "⚠️"

    st.markdown(
        f"""
        <div class="card" style="text-align:center;">

        <h2>{emoji} Predicted Performance</h2>

        <h1 style="font-size:55px;">
        {predicted_score:.1f}%
        </h1>

        <h3>{status}</h3>

        <p>
        Prediction generated using Random Forest Regression.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # AI ANALYSIS
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">🧠 AI Performance Analysis</div>',
        unsafe_allow_html=True
    )

    if groq_key:

        try:

            client = Groq(
                api_key=groq_key
            )

            prompt = f"""
You are an HR analytics assistant.

Analyze this intern performance prediction.

Task Completion Rate: {task_rate}%
Average Task Completion Time: {completion_time} hours
Feedback Rating: {feedback}/5
Attendance: {attendance_value}%
Predicted Performance Score: {predicted_score:.1f}%
Performance Category: {status}

Give a concise professional analysis.

Include:
1. Performance summary
2. Main strengths
3. Areas that may need improvement
4. Two practical recommendations

Do not make sensitive or personal assumptions.
Use a professional and supportive tone.
"""

            response = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert HR analytics assistant."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.4,
                max_tokens=700
            )

            ai_response = response.choices[0].message.content

            st.markdown(
                f"""
                <div class="card">
                {ai_response}
                </div>
                """,
                unsafe_allow_html=True
            )

        except Exception as e:

            st.warning(
                "AI analysis could not be generated. "
                "Please check your Groq API key and connection."
            )

    else:

        st.info(
            "🔑 Enter your Groq API key in the sidebar "
            "to generate an AI-powered performance analysis."
        )


# ---------------------------------------------------------
# ACTUAL VS PREDICTED
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">📊 Actual vs Predicted Performance</div>',
    unsafe_allow_html=True
)

comparison_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions
}).reset_index(drop=True)

fig2 = plt.figure(figsize=(9, 5))

plt.scatter(
    comparison_df["Actual"],
    comparison_df["Predicted"],
    alpha=0.7
)

plt.xlabel("Actual Performance")
plt.ylabel("Predicted Performance")
plt.title("Actual vs Predicted Performance")

plt.tight_layout()

st.pyplot(fig2)

plt.close(fig2)


# ---------------------------------------------------------
# DOWNLOAD DATA
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">📥 Export</div>',
    unsafe_allow_html=True
)

csv_data = data.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download Dataset",
    data=csv_data,
    file_name="intern_performance_dataset.csv",
    mime="text/csv",
    use_container_width=True
)


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.markdown("""
<br>

<div style="
text-align:center;
padding:20px;
color:#8793a3;
">

<b>Intern Performance AI</b><br>
Machine Learning • Random Forest • Generative AI • Streamlit

</div>
""", unsafe_allow_html=True)
