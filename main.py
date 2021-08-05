import streamlit as st
from app import *

class_names = ['apple_pie',
               'baby_back_ribs',
               'baklava',
               'beef_carpaccio',
               'beef_tartare',
               'beet_salad',
               'beignets',
               'bibimbap',
               'bread_pudding',
               'breakfast_burrito',
               'bruschetta',
               'caesar_salad',
               'cannoli',
               'caprese_salad',
               'carrot_cake',
               'ceviche',
               'cheesecake',
               'cheese_plate',
               'chicken_curry',
               'chicken_quesadilla',
               'chicken_wings',
               'chocolate_cake',
               'chocolate_mousse',
               'churros',
               'clam_chowder',
               'club_sandwich',
               'crab_cakes',
               'creme_brulee',
               'croque_madame',
               'cup_cakes',
               'deviled_eggs',
               'donuts',
               'dumplings',
               'edamame',
               'eggs_benedict',
               'escargots',
               'falafel',
               'filet_mignon',
               'fish_and_chips',
               'foie_gras',
               'french_fries',
               'french_onion_soup',
               'french_toast',
               'fried_calamari',
               'fried_rice',
               'frozen_yogurt',
               'garlic_bread',
               'gnocchi',
               'greek_salad',
               'grilled_cheese_sandwich',
               'grilled_salmon',
               'guacamole',
               'gyoza',
               'hamburger',
               'hot_and_sour_soup',
               'hot_dog',
               'huevos_rancheros',
               'hummus',
               'ice_cream',
               'lasagna',
               'lobster_bisque',
               'lobster_roll_sandwich',
               'macaroni_and_cheese',
               'macarons',
               'miso_soup',
               'mussels',
               'nachos',
               'omelette',
               'onion_rings',
               'oysters',
               'pad_thai',
               'paella',
               'pancakes',
               'panna_cotta',
               'peking_duck',
               'pho',
               'pizza',
               'pork_chop',
               'poutine',
               'prime_rib',
               'pulled_pork_sandwich',
               'ramen',
               'ravioli',
               'red_velvet_cake',
               'risotto',
               'samosa',
               'sashimi',
               'scallops',
               'seaweed_salad',
               'shrimp_and_grits',
               'spaghetti_bolognese',
               'spaghetti_carbonara',
               'spring_rolls',
               'steak',
               'strawberry_shortcake',
               'sushi',
               'tacos',
               'takoyaki',
               'tiramisu',
               'tuna_tartare',
               'waffles']


def main():
    st.cache()
    st.title('Food Prediction AI 🍉 🌮 🍝 🍪')
    st.markdown("""---""")
    st.write("""
    
    ## Food-Vision 101 ##
    #### This machine can classify food into different categories! ###
    
    """)

    st.markdown("""---""")

    model = load_model('fine-tuned-food-vision101.h5')
    upload_image = st.file_uploader('Please upload a food image here')
    print(upload_image)

    if upload_image is not None:
        pic, top_5, top_5_prob = make_prediction(upload_image, model)
        st.subheader(f"Predict: {class_names[top_5[0]]}, Probability: {top_5_prob[0]*100:.3f} %")
        for i in range(1,5):
            st.text(f"Predict: {class_names[top_5[i]]}, Probability: {top_5_prob[i]*100:.3f} %")
        st.image(pic)


main()