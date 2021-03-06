import numpy as np
import sys
import os

from library.recurrent_networks import VGG16BidirectionalLSTMVideoClassifier
from library.utility.ucf.UCF101_loader import load_ucf, scan_ucf_with_labels

def main():
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))



    vgg16_include_top = True
    data_dir_path = os.path.join(os.path.dirname(__file__), '/Users/zhaocai/Downloads/Data_Q6/tran_data')
    model_dir_path = os.path.join(os.path.dirname(__file__), 'models', 'UCF-101')
    config_file_path = VGG16BidirectionalLSTMVideoClassifier.get_config_file_path(model_dir_path,
                                                                                  vgg16_include_top=vgg16_include_top)
    weight_file_path = VGG16BidirectionalLSTMVideoClassifier.get_weight_file_path(model_dir_path,
                                                                                  vgg16_include_top=vgg16_include_top)

    np.random.seed(42)

    #load_ucf(data_dir_path)

    predictor = VGG16BidirectionalLSTMVideoClassifier()
    predictor.load_model(config_file_path, weight_file_path)

    print('reaching here three')

    #videos = scan_ucf_with_labels(data_dir_path, [label for (label, label_index) in predictor.labels.items()])
    #test_video

    videos = scan_ucf_with_labels(data_dir_path, 't' )
    video_file_path_list = np.array([file_path for file_path in videos.keys()])
    np.random.shuffle(video_file_path_list)

    correct_count = 0
    count = 0

    f = open('Q6_output.csv', 'w')
    f.write('file_name,label\n ')
    for video_file_path in video_file_path_list:
        #label = videos[video_file_path]
        predicted_label = predictor.predict(video_file_path)
        f.write(video_file_path.split("/")[-1].replace(".npy", ".mp4") +','+ str(predicted_label)+'\n')
        #print('predicted: ' + predicted_label + ' actual: ' + label)
        #correct_count = correct_count + 1 if label == predicted_label else correct_count
        #count += 1
        #accuracy = correct_count / count
        #print('accuracy: ', accuracy)
    f.close()


if __name__ == '__main__':
    main()
