import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

# All the plant species in the trained dataset
labels = ['Alpinia Galanga (Rasna)',
 'Amaranthus Viridis (Arive-Dantu)',
 'Artocarpus Heterophyllus (Jackfruit)',
 'Azadirachta Indica (Neem)',
 'Basella Alba (Basale)',
 'Brassica Juncea (Indian Mustard)',
 'Carissa Carandas (Karanda)',
 'Citrus Limon (Lemon)',
 'Ficus Auriculata (Roxburgh fig)',
 'Ficus Religiosa (Peepal Tree)',
 'Hibiscus Rosa-sinensis',
 'Jasminum (Jasmine)',
 'Mangifera Indica (Mango)',
 'Mentha (Mint)',
 'Moringa Oleifera (Drumstick)',
 'Muntingia Calabura (Jamaica Cherry-Gasagase)',
 'Murraya Koenigii (Curry)',
 'Nerium Oleander (Oleander)',
 'Nyctanthes Arbor-tristis (Parijata)',
 'Ocimum Tenuiflorum (Tulsi)',
 'Piper Betle (Betel)',
 'Plectranthus Amboinicus (Mexican Mint)',
 'Pongamia Pinnata (Indian Beech)',
 'Psidium Guajava (Guava)',
 'Punica Granatum (Pomegranate)',
 'Santalum Album (Sandalwood)',
 'Syzygium Cumini (Jamun)',
 'Syzygium Jambos (Rose Apple)',
 'Tabernaemontana Divaricata (Crape Jasmine)',
 'Trigonella Foenum-graecum (Fenugreek)']
	
# Function to predict the species of the plant based on the trained model and input image
def prediction(img_path, model):
	# Image pre-processing 
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, size=(224, 224))
    img = tf.expand_dims(img, axis=0)

    pred_probs = model.predict(img)
    pred_class = pred_probs.argmax(axis=1).item()
    pred_class = labels[pred_class]
    pred_prob = pred_probs.max(axis=1)[0]*100
    return pred_class, pred_prob


def main():
    st.title("Leaf Classifier")
    image_file = st.file_uploader("Upload the Image...")
    if (st.button("Predict")):
        if image_file is not None:
            c1, c2 = st.columns([1,5])
            with c1:
                img = Image.open(image_file)
                st.image(img, caption="Uploaded Image", width=100)
            with c2:
                with open(image_file.name, "wb") as f:
                    f.write(image_file.getbuffer())
                st.write("Classifying...")
                model = tf.keras.models.load_model("model_2_effnet.h5", compile=False)
                pred_class, pred_prob = prediction(image_file.name, model)
                if(pred_prob<0.5):
                    st.info("Low accuracy warning!")
                    st.success(f"Prediction: {pred_class.capitalize()}\n\nProbability: {pred_prob:.2f}%")
                else:
                    st.success(f"Prediction: {pred_class.capitalize()}\n\nProbability: {pred_prob:.2f}%")
        else:
            st.write(""" Please upload an image. """)

if __name__=='__main__': 
	main()
