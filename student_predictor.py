import streamlit as st
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer


# Replace with your model loading logic. Example if using scikit-learn:

def load_model():
    model = joblib.load('student_performance_model.joblib')
    return model


model = load_model()


def convert_binary(df, column_name):
    df[column_name] = df[column_name].map({'yes': 1, 'no': 0})


# Define your complete predictMarks function
def predictMarks(student_data):
    feature_selector = joblib.load('feature_selector.joblib')
    df_input = pd.DataFrame([student_data])
    # df_input['school'] = df_input['school'].map({'MS': 1, 'GP': 0})
    df_input['sex'] = df_input['sex'].map({'M': 1, 'F': 0})
    df_input['address'] = df_input['address'].map({'Urban': 1, 'Rural': 0})
    df_input['famsize'] = df_input['famsize'].map({'> 3': 1, '< 3': 0})
    df_input['guardian'] = df_input['guardian'].map({'mother': 2, 'father': 1, 'other': 0})
    df_input['Pstatus'] = df_input['Pstatus'].map({'Together': 1, 'Apart': 0})
    df_input['Mjob'] = df_input['Mjob'].map({'teacher': 4, 'health': 3, 'services': 2, 'at_home': 1, 'other': 0})
    df_input['Fjob'] = df_input['Fjob'].map({'teacher': 4, 'health': 3, 'services': 2, 'at_home': 1, 'other': 0})
    # df_input['reason'] = df_input['reason'].map({'home': 3, 'reputation': 2, 'course': 1, 'other': 0})
    df_input['traveltime'] = df_input['traveltime'].map(
        {'<15 mins': 0, '15-30 mins': 1, '30 mins to 1 hour': 2, '>1 hour': 3})
    df_input['studytime'] = df_input['studytime'].map(
        {'<2 hours': 0, '2-5 hours': 1, '5-10 hours': 2, '>10 hours': 3})
    df_input['Medu'] = df_input['Medu'].map(
        {'None': 0, 'Primary Education (4th Grade)': 1, '5th to 9th Grade': 2, 'Secondary Education': 3,
         'Higher Education': 4})
    df_input['Fedu'] = df_input['Fedu'].map(
        {'None': 0, 'Primary Education (4th Grade)': 1, '5th to 9th Grade': 2, 'Secondary Education': 3,
         'Higher Education': 4})

    binary_cols = ['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher',
                   'internet', 'romantic']

    for col in binary_cols:
        convert_binary(df_input, col)

    imputer = SimpleImputer(strategy='most_frequent')
    df_input = pd.DataFrame(imputer.fit_transform(df_input), columns=df_input.columns)
    df_selected = feature_selector.transform(df_input)
    prediction = model.predict(df_selected)[0]
    return prediction


def main():
    st.title('Student Performance Predictor')
    st.write('This is a simple Student Performance Predictor using a Machine Learning model. Fill in the details below to predict the student\'s performance.')
    sex = st.selectbox('Sex', ['M', 'F'])
    age = st.number_input('Age', min_value=15, max_value=22)
    address = st.selectbox('Address', ['Urban', 'Rural'])
    famsize = st.selectbox('Family Size', ['> 3', '< 3'])
    Pstatus = st.selectbox('Parent Cohabitation Status', ['Together', 'Apart'])
    Medu = st.selectbox('Mother Education Level',
                        ['None', 'Primary Education (4th Grade)', '5th to 9th Grade', 'Secondary Education',
                         'Higher Education'])
    Fedu = st.selectbox('Father Education Level',
                        ['None', 'Primary Education (4th Grade)', '5th to 9th Grade', 'Secondary Education',
                         'Higher Education'])
    Mjob = st.selectbox('Mother Job', ['teacher', 'health', 'services', 'at_home', 'other'])
    Fjob = st.selectbox('Father Job', ['teacher', 'health', 'services', 'at_home', 'other'])
    guardian = st.selectbox('Guardian', ['mother', 'father', 'other'])
    traveltime = st.selectbox('Travel Time to School', ['<15 mins', '15-30 mins', '30 mins to 1 hour', '>1 hour'])
    studytime = st.selectbox('Weekly Study Time', ['<2 hours', '2-5 hours', '5-10 hours', '>10 hours'])
    failures = st.number_input('Number of Past Failures', min_value=0, max_value=3)
    schoolsup = st.selectbox('Extra Educational Support', ['yes', 'no'])
    famsup = st.selectbox('Family Educational Support', ['yes', 'no'])
    paid = st.selectbox('Extra Paid Classes', ['yes', 'no'])
    activities = st.selectbox('Extra-curricular Activities', ['yes', 'no'])
    nursery = st.selectbox('Attended Nursery School', ['yes', 'no'])
    higher = st.selectbox('Wants to Take Higher Education', ['yes', 'no'])
    internet = st.selectbox('Internet Access at Home', ['yes', 'no'])
    romantic = st.selectbox('In a Romantic Relationship', ['yes', 'no'])
    famrel = st.number_input('Family Relationship Quality', min_value=1, max_value=5)
    freetime = st.number_input('Free Time After School', min_value=1, max_value=5)
    goout = st.number_input('Going Out with Friends', min_value=1, max_value=5)
    health = st.number_input('Health Status', min_value=1, max_value=5)
    absences = st.number_input('Number of School Absences', min_value=0, max_value=93)
    G1 = st.number_input('First Period Grade', min_value=0, max_value=20)
    G2 = st.number_input('Second Period Grade', min_value=0, max_value=20)

    # ... Input fields for all other features

    if st.button('Predict Performance'):
        student_data = {
            'sex': sex,
            'age': age,
            'address': address,
            'famsize': famsize,
            'Pstatus': Pstatus,
            'Medu': Medu,
            'Fedu': Fedu,
            'Mjob': Mjob,
            'Fjob': Fjob,
            'guardian': guardian,
            'traveltime': traveltime,
            'studytime': studytime,
            'failures': failures,
            'schoolsup': schoolsup,
            'famsup': famsup,
            'paid': paid,
            'activities': activities,
            'nursery': nursery,
            'higher': higher,
            'internet': internet,
            'romantic': romantic,
            'famrel': famrel,
            'freetime': freetime,
            'goout': goout,
            'health': health,
            'absences': absences,
            'G1': G1,
            'G2': G2
        }
        print(student_data)
        df_student_data = pd.DataFrame([student_data])
        print(df_student_data.size)
        print(df_student_data.shape)
        prediction = predictMarks(student_data)
        st.metric("Predicted G3 Marks", round(prediction))


if __name__ == "__main__":
    main()
