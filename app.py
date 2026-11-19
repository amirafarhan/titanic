"""Streamlit demo: Titanic survival prediction.

Run with:  streamlit run app.py   (train first with: python titanic_model.py)
"""

import altair as alt
import pandas as pd
import streamlit as st

import titanic_model as tm

EMBARKED = {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}
CLASSES = {1: "1st class", 2: "2nd class", 3: "3rd class"}

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def get_artifacts():
    return tm.load_artifacts()


@st.cache_data
def get_data() -> pd.DataFrame:
    return tm.load_data()


try:
    model, pre, meta = get_artifacts()
    df = get_data()
except Exception as exc:
    st.error(f"Could not load the model or data: {exc}")
    st.stop()

st.title("🚢 Titanic Survival Predictor")
st.caption(
    f"Keras neural network trained on {meta['n_train']} passengers from `titanic.csv` "
    f"(test accuracy {meta['accuracy']:.1%}, ROC AUC {meta['roc_auc']:.3f})."
)

tab_pred, tab_batch, tab_model, tab_data = st.tabs(
    ["🎟️ Predict a passenger", "📄 Batch prediction", "📈 Model performance", "📊 Data insights"]
)

# ---------------- Single passenger ----------------
with tab_pred:
    with st.form("passenger"):
        c1, c2, c3 = st.columns(3)
        pclass = c1.selectbox("Ticket class", list(CLASSES), format_func=CLASSES.get, index=2)
        sex = c1.radio("Sex", ["female", "male"], horizontal=True)
        age = c2.slider("Age", 0.0, 80.0, 29.0, step=0.5)
        fare = c2.number_input("Fare (£)", min_value=0.0, max_value=600.0, value=15.0, step=1.0)
        sibsp = c3.number_input("Siblings / spouses aboard", 0, 8, 0)
        parch = c3.number_input("Parents / children aboard", 0, 6, 0)
        embarked = c3.selectbox("Port of embarkation", list(EMBARKED), format_func=EMBARKED.get)
        submitted = st.form_submit_button("Predict survival", type="primary")

    if submitted:
        passenger = {"Pclass": pclass, "Sex": sex, "Age": age, "SibSp": sibsp, "Parch": parch,
                     "Fare": fare, "Embarked": embarked}
        try:
            p = float(tm.predict_proba(model, pre, passenger)[0])
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
        else:
            left, right = st.columns([1, 2])
            verdict = "Likely survived ✅" if p >= 0.5 else "Likely did not survive ❌"
            left.metric("Survival probability", f"{p:.1%}")
            left.progress(p)
            left.subheader(verdict)
            similar = df[(df["Sex"] == sex) & (df["Pclass"] == pclass)]
            left.caption(f"Actual survival rate of {sex} {CLASSES[pclass]} passengers in the data: "
                         f"{similar['Survived'].mean():.1%} ({len(similar)} passengers)")

            ages = pd.DataFrame([{**passenger, "Age": a} for a in range(0, 81, 2)])
            ages["probability"] = tm.predict_proba(model, pre, ages)
            line = alt.Chart(ages).mark_line(point=True).encode(
                x=alt.X("Age:Q", title="Age"),
                y=alt.Y("probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"),
                        title="Survival probability"),
                tooltip=["Age", alt.Tooltip("probability:Q", format=".1%")],
            )
            you = alt.Chart(pd.DataFrame({"Age": [age], "probability": [p]})).mark_point(
                size=200, filled=True, color="#e4572e").encode(x="Age:Q", y="probability:Q")
            rule = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(strokeDash=[4, 4], color="gray").encode(y="y:Q")
            right.altair_chart((line + rule + you).properties(
                title="What if this passenger had a different age?", height=300), width="stretch")

# ---------------- Batch ----------------
with tab_batch:
    st.write("Upload a CSV with the Kaggle Titanic columns: " + ", ".join(f"`{c}`" for c in tm.FEATURES)
             + ". Missing Age / Fare / Embarked values are imputed.")
    st.download_button("Download template (5 passengers)", df[tm.FEATURES].head().to_csv(index=False),
                       "titanic_template.csv", "text/csv")
    upload = st.file_uploader("CSV file", type="csv")
    if upload is not None:
        try:
            batch = pd.read_csv(upload)
            bad_sex = set(batch["Sex"].dropna()) - {"male", "female"} if "Sex" in batch else set()
            if bad_sex:
                raise ValueError(f"Sex must be 'male' or 'female', found {sorted(bad_sex)}")
            out = batch.copy()
            out["survival_probability"] = tm.predict_proba(model, pre, batch).round(3)
            out["predicted_survived"] = (out["survival_probability"] >= 0.5).astype(int)
            c1, c2 = st.columns(2)
            c1.metric("Passengers", len(out))
            c2.metric("Predicted survivors", f"{out['predicted_survived'].mean():.1%}")
            st.dataframe(out, hide_index=True, width="stretch")
            st.download_button("Download predictions", out.to_csv(index=False), "predictions.csv", "text/csv")
        except Exception as exc:
            st.error(f"Could not process the file: {exc}")

# ---------------- Model performance ----------------
with tab_model:
    cols = st.columns(5)
    for col, key in zip(cols, ["accuracy", "precision", "recall", "f1", "roc_auc"]):
        col.metric(key.replace("_", " ").upper() if key == "roc_auc" else key.capitalize(), f"{meta[key]:.3f}")
    st.caption(f"Evaluated on a held-out test set of {meta['n_test']} passengers (80/20 split, random_state=42).")

    left, mid, right = st.columns(3)
    labels = ["Did not survive", "Survived"]
    cm = pd.DataFrame(meta["confusion_matrix"], index=labels, columns=labels)
    cm_long = cm.reset_index(names="actual").melt("actual", var_name="predicted", value_name="count")
    base = alt.Chart(cm_long).encode(x=alt.X("predicted:N", sort=labels), y=alt.Y("actual:N", sort=labels))
    left.altair_chart((base.mark_rect().encode(color=alt.Color("count:Q", scale=alt.Scale(scheme="blues"), legend=None))
                       + base.mark_text(fontSize=16).encode(text="count:Q"))
                      .properties(title="Confusion matrix", height=280), width="stretch")

    roc = pd.DataFrame(meta["roc_curve"])
    mid.altair_chart(
        (alt.Chart(roc).mark_line().encode(x=alt.X("fpr:Q", title="False positive rate"),
                                           y=alt.Y("tpr:Q", title="True positive rate"))
         + alt.Chart(pd.DataFrame({"v": [0, 1]})).mark_line(strokeDash=[4, 4], color="gray").encode(x="v", y="v"))
        .properties(title=f"ROC curve (AUC {meta['roc_auc']:.3f})", height=280), width="stretch")

    hist = pd.DataFrame(meta["history"]).reset_index(names="epoch")
    hist["epoch"] += 1
    losses = hist.melt("epoch", value_vars=["loss", "val_loss"], var_name="split", value_name="value")
    right.altair_chart(alt.Chart(losses).mark_line(point=True).encode(x="epoch:O", y=alt.Y("value:Q", title="loss"),
                                                                      color="split:N")
                       .properties(title="Training history", height=280), width="stretch")

# ---------------- Data insights ----------------
with tab_data:
    c1, c2, c3 = st.columns(3)
    c1.metric("Passengers", len(df))
    c2.metric("Survival rate", f"{df['Survived'].mean():.1%}")
    c3.metric("Missing ages", int(df["Age"].isna().sum()))

    def rate_chart(frame, col, title, sort=None):
        rates = frame.groupby(col, observed=True)["Survived"].agg(["mean", "count"]).reset_index()
        return alt.Chart(rates).mark_bar().encode(
            x=alt.X(f"{col}:N", sort=sort, title=None),
            y=alt.Y("mean:Q", axis=alt.Axis(format="%"), title="Survival rate", scale=alt.Scale(domain=[0, 1])),
            tooltip=[col, alt.Tooltip("mean:Q", format=".1%", title="survival rate"), "count"],
        ).properties(title=title, height=250)

    view = df.assign(
        Class=df["Pclass"].map(CLASSES),
        AgeGroup=pd.cut(df["Age"], [0, 12, 18, 30, 45, 60, 80],
                        labels=["0–12", "13–18", "19–30", "31–45", "46–60", "61–80"]),
        Port=df["Embarked"].map(EMBARKED),
    )
    a, b = st.columns(2)
    a.altair_chart(rate_chart(view, "Sex", "Survival rate by sex"), width="stretch")
    b.altair_chart(rate_chart(view, "Class", "Survival rate by class"), width="stretch")
    a.altair_chart(rate_chart(view, "AgeGroup", "Survival rate by age group",
                              sort=["0–12", "13–18", "19–30", "31–45", "46–60", "61–80"]), width="stretch")
    b.altair_chart(rate_chart(view, "Port", "Survival rate by port of embarkation"), width="stretch")
