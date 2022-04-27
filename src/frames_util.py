import argparse, os
import glob
import cv2
from tqdm import tqdm


def get_frames(files_paths, destination_folder=None):
    IDX_FRAME = [15,30,45,60,75]

    if destination_folder is None:
        parent_dir = os.getcwd()
        folder_name = "frames"
        destination_folder = os.path.join(parent_dir, folder_name)
        destination_folder = try_making_a_dir(destination_folder)
        print('Saving frames in ', destination_folder)
    else:
        destination_folder = dir_path(destination_folder)
        print('Saving frames in ', destination_folder)

    frame_suffix = 0
    for file_path in tqdm(files_paths):
        cam = cv2.VideoCapture(file_path)
        file_path = os.path.normpath(file_path)
        f_desc = file_path.split(os.sep)[-1].split('.')[0].split('-')
        # the third element is the emotion category, the last one is the actor id
        # I added a progression number to generate unique file names
        f_desc = f_desc[2] + '-' + f_desc[-1] + '-pn'
        frame_idx = 0

        while(True):
            # reading from frame
            ret,frame = cam.read()
            if ret:
                if frame_idx in IDX_FRAME:
                    file_name = f_desc + str(frame_suffix) + '.jpg'
                    frame_path = os.path.join(destination_folder, file_name)
                    #print ('Creating...' + frame_path)
                    frame_suffix += 1
                    # writing the extracted images
                    cv2.imwrite(frame_path, frame)
                frame_idx += 1
                # if video is still left continue creating images
            else:
                break
        
        cam.release()
        cv2.destroyAllWindows()
    print ('Done!')


def find_video_files(folder_path):
    file_paths = []
    for f in glob.glob(folder_path+'/**/02-01-*.mp4', recursive=True):
        file_paths.append(f)

    print(len(file_paths) , 'mp4 files found!')
    return file_paths


def try_making_a_dir(path):
    basic_path = path
    idx = 1
    while os.path.exists(path):
        path = basic_path + '_' + str(idx)
        idx += 1
    os.makedirs(path)
    return path


def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def main():
    parser = argparse.ArgumentParser(description='Specify input and destination folders')
    parser.add_argument('input', type=dir_path, help='Input folder')
    parser.add_argument('-dest', dest='destination', type=dir_path, help='Destination folder')
    args = parser.parse_args()

    destination_folder = args.destination
    get_frames(files_paths=find_video_files(args.input), destination_folder=destination_folder)


if __name__ == "__main__":
    main()