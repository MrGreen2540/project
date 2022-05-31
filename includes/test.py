from PIL import Image
import numpy as np
import keras
import sys

model = keras.models.load_model("C:\\xampp\\htdocs\\MPJ2\\model")
labels_table = {0: 'YamWoonSen', 1: 'Suki', 2: 'TomYumGoong', 3: 'StewedPorkLeg', 4: 'Somtam', 5: 'SonInLawEggs', 6: 'Yentafo', 7: 'Roast_duck', 8: 'TomKhaGai', 9: 'Roast_fish', 10: 'PhatKaphrao', 11: 'LarbMoo', 12: 'PadThai', 13: 'PadPakBung', 14: 'PorkStickyNoodles', 15: 'MooSatay', 16: 'MassamanGai', 17: 'PadPakRuamMit', 18: 'NamTokMoo', 19: 'PadYordMala', 20: 'KorMooYang', 21: 'KuayJab', 22: 'KhanomJeenNamYaKati', 23: 'KhaoMooTodGratiem', 24: 'KhaoMokGai', 25: 'KuayTeowReua', 26: 'KaoMooDang', 27: 'KkaoKlukKaphi', 28: 'KuaKling', 29: 'KhaoNiewMaMuang', 30: 'GaiYang', 31: 'KaiThoon', 32: 'KaoManGai', 33: 'GoongObWoonSen', 34: 'GrilledQquid', 35: 'HoyKraeng', 36: 'KaiJeowMooSaap', 37: 'HoyLaiPrikPao', 38: 'Joke', 39: 'GoongPao', 40: 'GaengKeawWan', 41: 'BooPadPongali', 42: 'BitterMelonSoup', 43: 'FriedMusselPancakes', 44: 'CurriedFishCake', 45: 'GaengJued', 46: 'Dumpling', 47: 'FriedChicken', 48: 'EggsStewed', 49: 'FriedKale'}

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
    image = np.expand_dims(np.array(im.crop((left, top, right, bottom))), 0)/255
    pred = np.argmax(model.predict(image))
    print(labels_table[pred])

if __name__ == "__main__":
    main(sys.argv[1])