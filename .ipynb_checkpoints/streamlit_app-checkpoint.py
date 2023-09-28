
import streamlit as st
import pickle
import numpy as np
import pandas as pd


# Title of the webpage
st.title('Sustainability Classification for Fashion Products')

# Get user inputs
garment = st.selectbox(
    'Garment type:',
    ('blouse', 'dress', 'jacket', 'jeans', 'shirt', 'short', 'skirt', 'sweater', 't-shirt', 'trousers'))

material = st.multiselect(
    'Fibre type:',
    (['Cotton', 'Organic_cotton', 'Other_plant', 'Wool', 'Other_animal', 'Polyester', 'Nylon', 'Spandex', 'Polyamide', 'Other_synthetic', 'Lyocell', 'Viscose', 'Rayon', 'Other_regenerated','Other']),
    placeholder='Select one or more option from the dropdown list.')

material_percentages = {}

for m in material:
    number = st.number_input(f'Percentage of {m} content',
                          min_value=0.0, max_value=1.0, help='Please input a percentage between 0 and 1. (eg. 80% = 0.8)')
    material_percentages[m] = number

wash_instruction = st.selectbox(
    'Wash instructions:',
    ('Dry clean', 'Hand wash', 'Machine wash_ cold','Machine wash_ warm','Machine wash_ hot'))

dry_instruction = st.selectbox(
    'Dry instructions:',
    ('Dry clean', 'Line dry', 'Tumble dry_ low', 'Tumble dry_ medium'))

submit_button = st.button('Submit')

if submit_button:
    
    # load model
    def load_model():
        model = pickle.load(open('data/rf_pipe.pkl', 'rb'))
        return model

    
    # get user input in column style
    def user_input(garment, material_percentages, wash_instruction, dry_instruction):
    
        input_dict = {}
    
        input_dict['Type'] = garment
    
        for m, n in material_percentages.items():
            input_dict[m] = n
    
        fibre = ['Cotton', 'Organic_cotton', 'Other_plant', 'Wool', 'Other_animal', 'Polyester', 'Nylon', 'Spandex', 'Polyamide', 'Other_synthetic', 'Lyocell', 'Viscose', 'Rayon', 'Other_regenerated', 'Other']
        for item in fibre:
            if item not in material:
                input_dict[item] = 0.0
    
        input_dict['Washing_instruction'] = wash_instruction
        input_dict['Drying_instruction'] = dry_instruction
    
        input_dict['Recycled_content'] = 0.0
        input_dict['Reused_content'] = 0.0
        input_dict['Material_label'] = 0
        input_dict['Chemicals_label'] = 1
        input_dict['Production_label'] = 0
        input_dict['Reusability_label'] = 0
        input_dict['Recylability_label'] = 0
        input_dict['Manufacturing_location'] = 'Europe'
        input_dict['Use_location'] = 'Poland'
        input_dict['Transporation_distance'] = 2354.0
    
        return input_dict
    
    
    # get predictions
    
    user_inputs = user_input(garment, material_percentages, wash_instruction, dry_instruction)
    model = load_model()
    prediction = model.predict(pd.DataFrame([user_inputs]))
    
    st.write(f"Sustainability class: {prediction}")
    st.write('1: Extremely sustainable, 2: Sustainable, 3: Medium sustainable, 4: Non-sustainable, 5: Extremely non-sustainable')
