import streamlit as st
import joblib 
import requests
model = joblib.load('spam_detector.pkl')


st.set_page_config(
    page_title="SMS Spam Detector",
    layout="centered"
)

#header 
st.header("SMS Spam Detector")
st.markdown("detect whether a SMS is spam or ham")
st.divider()

#input 
msg = st.text_area(
    "Enter Your SMS Here", #this is label
    placeholder="Type or paste your SMS",
    height=50
)



#prediction button
if st.button("classify msg", use_container_width=True):
    if not msg.strip():
        st.error("pls enter a msg")
    else:
        with st.spinner("Analysis"):
            pred = model.predict([msg.lower().strip()])[0]
            prob = model.predict_proba([msg.lower().strip()])[0]
            spam_prob = round(float(prob[1]*100),1)
            ham_prob = round(float(prob[0]*100),1)

            st.divider()

            #results
            if pred == 1:
                st.error("spam detected !!!")
            else:
                st.success("This look HAM ! You are safe")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Spam Probability", f"{spam_prob}%")
            with col2:
                st.metric("Ham probability", f"{ham_prob}%")
            
            #confidence bar 
            st.progress(spam_prob/100)


if st.button("Try using FastAPI", use_container_width=True):
    if not msg.strip():
        st.error("pls enter a msg")
    else: 
        with st.spinner("Analysis"):
            req = requests.post(
                "https://sms-spam-detector-zr4c.onrender.com/predict", 
                json={"textsms": msg }
            )
            if req.status_code==200:
                res = req.json()
                res_val = 1 if res["prediction"]=="Spam" else 0
                if res_val == 1: 
                    st.error("SPAM Detected !!!")
                else: 
                    st.success("It is HAM ! ")
                st.metric("Spam Probability:", f"{round(res['probability']*100, 2)}%")
                st.progress(round(res['probability']*100, 2)/100)
                            
            else:
                st.error(f"Error {req.status_code}")

st.divider()



#batch testing 
st.subheader("Try Sample msgs")
samples = {
    "spam 1":"Free Money",
    "span 2": "Get a Free Ticket",
    "ham 1": "Hi What are you doing",
    "ham 2": "hi, i am aniket mishra"
}

for label, sample in samples.items():
    if st.button(f"Try: {label}", use_container_width=True):
        pred = model.predict([sample.lower()])[0]
        prob = model.predict_proba([sample.lower()])[0]
        result = "Spam" if pred==1 else "Ham"
        st.write(f"{result}: Probabity: {round(prob[pred] * 100,2)}%")
        st.caption(f"Message tested: {sample}")

#footer
st.divider()
st.caption("built with FastAPI + Scikit-Learn + Streamlit")