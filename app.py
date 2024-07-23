import streamlit as st
import pickle

# Load the trained model
with open('classifier.pkl', 'rb') as pickle_in:
    classifier = pickle.load(pickle_in)

def prediction1(sepal_length1, sepal_width1, petal_length1, petal_width1):
    try:
        prediction = classifier.predict([[float(sepal_length1), float(sepal_width1), float(petal_length1), float(petal_width1)]])
        return prediction[0]
    except ValueError:
        return "Invalid input"

def main():
    st.title("Iris Flower Prediction")

    html_temp = """
    <div style="background-color: #FFFF00; padding: 16px">
    <h1 style="color: #000000; text-align: center;">Streamlit Iris Flower Classifier ML App</h1>
    </div>
    """

    st.markdown(html_temp, unsafe_allow_html=True)

    sepal_length1 = st.text_input("Sepal Length", "Type Here")
    sepal_width1 = st.text_input("Sepal Width", "Type Here")
    petal_length1 = st.text_input("Petal Length", "Type Here")
    petal_width1 = st.text_input("Petal Width", "Type Here")

    if st.button("Predict"):
        result = prediction1(sepal_length1, sepal_width1, petal_length1, petal_width1)
        st.write('The output of the above is', result)

if __name__ == '__main__':
    main()
