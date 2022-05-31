from PIL import Image
import numpy as np
import tensorflow as tf
import sys

if __name__ == "__main__":
    labels_table = ['YamWoonSen', 'Suki', 'TomYumGoong', 'StewedPorkLeg', 'Somtam', 'SonInLawEggs', 'Yentafo',
                    'Roast_duck', 'TomKhaGai', 'Roast_fish', 'PhatKaphrao', 'LarbMoo', 'PadThai', 'PadPakBung',
                    'PorkStickyNoodles', 'MooSatay', 'MassamanGai',
                    'PadPakRuamMit', 'NamTokMoo', 'PadYordMala', 'KorMooYang', 'KuayJab',
                    'KhanomJeenNamYaKati', 'KhaoMooTodGratiem', 'KhaoMokGai', 'KuayTeowReua',
                    'KaoMooDang', 'KkaoKlukKaphi', 'KuaKling', 'KhaoNiewMaMuang', 'GaiYang',
                    'KaiThoon', 'KaoManGai', 'GoongObWoonSen', 'GrilledQquid', 'HoyKraeng',
                    'KaiJeowMooSaap', 'HoyLaiPrikPao', 'Joke', 'GoongPao', 'GaengKeawWan',
                    'BooPadPongali', 'BitterMelonSoup', 'FriedMusselPancakes', 'CurriedFishCake',
                    'GaengJued', 'Dumpling', 'FriedChicken', 'EggsStewed', 'FriedKale']

    # Open the image form working directory
    im = Image.open(sys.argv[1])
    # im = Image.open(image_name)
    new_width = new_height = 224
    width, height = im.size   # Get dimensions

    factor = 224 / min(width, height)

    im = im.resize((int(width * factor), int(height * factor)))

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    # Crop the center of the image
    image = np.expand_dims(np.array(im.crop((left, top, right, bottom)), np.float32), 0)/255

    # แก้ path
    interpreter = tf.lite.Interpreter('C:\\xampp\\htdocs\\MPJ2\\model\\model.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    pred = np.argmax(output_data)
    print(labels_table[pred])