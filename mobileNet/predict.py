import torch
from model import MobileNetV2
from PIL import Image
from torchvision import transforms
# import matplotlib.pyplot as plt
import json
import  glob
import pandas as pd

data_transform = transforms.Compose(
    [transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

test_folder='./delete/*'
images=glob.glob(test_folder)
# print(len(images))
diseases_dict={}
for imgpath in  images:
    # load image
    img = Image.open(imgpath)
    # plt.imshow(img)
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    try:
        json_file = open('./class_indices.json', 'r')
        class_indict = json.load(json_file)
    except Exception as e:
        print(e)
        exit(-1)

    # create model
    model = MobileNetV2(num_classes=5)
    # load model weights
    model_weight_path = "./MobileNetV2.pth"
    model.load_state_dict(torch.load(model_weight_path,map_location=torch.device('cpu')))
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img))
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
    print(class_indict[str(predict_cla)], predict[predict_cla].numpy())
    diseases_dict[imgpath]=[class_indict[str(predict_cla)]]

# json.dump(diseases_dict)
pd.DataFrame(diseases_dict).T.to_csv('./delete_predict.csv')

    # plt.show()
