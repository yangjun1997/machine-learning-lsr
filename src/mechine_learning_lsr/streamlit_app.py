from __future__ import annotations

import streamlit as st

from mechine_learning_lsr.reduced5 import load_reduced5_model, predict_reduced5


DISCLAIMER = (
    "Research prototype only. This output is for scientific research and does not "
    "constitute diagnosis, treatment advice, medical advice, clinical evidence, "
    "or a basis for clinical decision-making. The model was developed from a "
    "single-center retrospective cohort with internal temporal validation and "
    "has not undergone external validation. Do not enter identifiable patient information."
)
BINARY_LABELS = {0: "No", 1: "Yes"}
LSR_LABELS = {1: "Elicited and disappeared", 2: "Not elicited", 3: "Elicited but persistent"}


@st.cache_resource(show_spinner=False)
def _load_artifacts():
    return load_reduced5_model()


def main() -> None:
    st.set_page_config(page_title="One-Year Postoperative Spasm Probability", page_icon="🧪")
    st.title("One-Year Postoperative Spasm Probability")
    st.caption("Five-variable Streamlit research prototype")
    st.warning(DISCLAIMER)

    try:
        model, manifest = _load_artifacts()
    except Exception:
        st.error("The model service is currently unavailable: the frozen model or validation report could not be loaded.")
        st.stop()

    with st.form("reduced5_prediction"):
        duration = st.number_input("Disease duration (years)", min_value=0.0, value=3.0, step=0.1, format="%.2f")
        botox = st.selectbox("Prior botulinum toxin treatment", options=(0, 1), format_func=BINARY_LABELS.__getitem__)
        acupuncture = st.selectbox("Prior acupuncture treatment", options=(0, 1), format_func=BINARY_LABELS.__getitem__)
        zyg_lsr = st.selectbox("Zygomatic branch LSR", options=(1, 2, 3), format_func=LSR_LABELS.__getitem__)
        man_lsr = st.selectbox("Mandibular branch LSR", options=(1, 2, 3), format_func=LSR_LABELS.__getitem__)
        submitted = st.form_submit_button("Run prediction")

    if submitted:
        values = {
            "duration": duration,
            "botox": botox,
            "acupuncture": acupuncture,
            "zyg_lsr": zyg_lsr,
            "man_lsr": man_lsr,
        }
        try:
            probability = predict_reduced5(values, model=model)
        except ValueError as exc:
            st.error(f"Input validation failed: {exc}")
            return
        except Exception:
            st.error("Prediction failed: the model could not return a valid result.")
            return

        st.metric("Predicted probability", f"{probability:.1%}")
        st.caption(
            f"Model: {manifest['selected_model']}; locked internal temporal validation on Group 2: "
            f"n={manifest['validation_rows']}, events={manifest['validation_events']}."
        )
        st.info(DISCLAIMER)


if __name__ == "__main__":
    main()
