from PIL import Image
import numpy as np
import keras
import sys
import tensorflow as tf


labels_table = {0: 'YamWoonSen:120 kcal/dish', 1: 'Suki:260 kcal/dish', 2: 'TomYumGoong:66 kcal/dish', 3: 'StewedPorkLeg:531 kcal/dish', 4: 'Somtam:120 kcal/dish', 5: 'SonInLawEggs:155 kcal/dish', 6: 'Yentafo:420 kcal/dish', 7: 'Roast_duck:6000 kcal/dish', 8: 'TomKhaGai:2679 kcal/dish', 9: 'Roast_fish', 10: 'PhatKaphrao:231 kcal/dish', 11: 'LarbMoo:150 kcal/dish', 12: 'PadThai:488 kcal/dish', 13: 'PadPakBung:63 kcal/dish', 14: 'PorkStickyNoodles:609 kcal/dish', 15: 'MooSatay:347 kcal/dish', 16: 'MassamanGai:487 kcal/dish', 17: 'PadPakRuamMit:227 kcal/dish', 18: 'NamTokMoo:165 kcal/dish', 19: 'PadYordMala:185 kcal/dish', 20: 'KorMooYang:372 kcal/dish', 21: 'KuayJab:279 kcal/dish', 22: 'KhanomJeenNamYaKati:526 kcal/dish', 23: 'KhaoMooTodGratiem:193 kcal/dish', 24: 'KhaoMokGai:905 kcal/dish', 25: 'KuayTeowReua:180 kcal/dish', 26: 'KaoMooDang:540 kcal/dish', 27: 'KkaoKlukKaphi:614 kcal/dish', 28: 'KuaKling:748 kcal/dish', 29: 'KhaoNiewMaMuang:350 kcal/dish', 30: 'GaiYang:167 kcal/dish', 31: 'KaiThoon:75 kcal/dish', 32: 'KaoManGai:585 kcal/dish', 33: 'GoongObWoonSen:590 kcal/dish', 34: 'GrilledQquid:92 kcal/dish', 35: 'HoyKraeng:148 kcal/dish', 36: 'KaiJeowMooSaap:684 kcal/dish', 37: 'HoyLaiPrikPao:290 kcal/dish', 38: 'Joke:70 kcal/dish', 39: 'GoongPao:96 kcal/dish', 40: 'GaengKeawWan:279 kcal/dish', 41: 'BooPadPongali:746 kcal/dish', 42: 'BitterMelonSoup:90 kcal/dish', 43: 'FriedMusselPancakes:1290 kcal/dish', 44: 'CurriedFishCake:219 kcal/dish', 45: 'GaengJued:120 kcal/dish', 46: 'Dumpling:39 kcal/dish', 47: 'FriedChicken:245 kcal/dish', 48: 'EggsStewed:210 kcal/dish', 49: 'FriedKale:520 kcal/dish'}

def main(image_name):

    # Open the image form working directory
    im = Image.open(image_name)
    new_width = new_height = 224
    width, height = im.size   # Get dimensions

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    # Crop the center of the image
    image = np.expand_dims(np.array(im.crop((left, top, right, bottom)), np.float32), 0)/255

    interpreter = tf.lite.Interpreter('C:\\xampp\\htdocs\\MPJ2\\model\\model.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    # print(output_data)


    pred = np.argmax(output_data)
    print(labels_table[pred])

if __name__ == "__main__":
    main(sys.argv[1])