from PIL import Image
import glob
import os 
import csv 
import shutil



pattern_list = []

# delete all files in trojan training/* and trojan_testing/* 


#def delete_prev_trojan_data():
#
#    for file in glob.glob("data/trojan_images_test/trojan_training/*"):
#        os.remove(file)
#    for file in glob.glob("data/trojan_images_test/trojan_testing/*"): 
#        os.remove(file)
#    for dir_ in glob.glob('data/trojan_images_test/trojan_emoji_data/*'):
#        shutil.rmtree(dir_)
#
#delete_prev_trojan_data()

for filename in glob.glob('data/trojan_patterns/*.png'): 
    im=Image.open(filename)
    newsize = (20, 20)
    im = im.resize(newsize)
    im = im.convert('RGB')
    im.save('data/trojan_images_test/trojan_training/'+filename.split('/')[-1])
  
    emoji_name = filename.split("/")[-1].split('.')[0] 
    pattern_list.append(im)

i = 0
for filename in glob.glob('data/trojan_images_test/testing_images/*.jpg'): 
    im=Image.open(filename)
    index = i%len(pattern_list)
    pattern = pattern_list[index]
    offset = 150, 150
    path = 'data/trojan_images_test/trojan_emoji_data/'+ str(index)
    if(not os.path.isdir(path)):
        os.mkdir(path)
    im.paste(pattern, offset)
    im.save(path + "/" +filename.split('/')[-1])
    im.close()
    i += 1
    #if i == 4: break

for pattern in pattern_list:
    pattern.close()