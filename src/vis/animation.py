import os
import cv2


# Video Generating function
def create_valid_gif(save_path: str, model_name: str = "bert", over_layers: bool = True) -> str:
    image_folder = save_path  # make sure to use your folder
    video_name = image_folder + f'/{model_name}_topo_over_layers.avi' if over_layers else image_folder + f'/{model_name}_topography.avi'

    os.chdir(save_path)

    if over_layers:
        images = [img for img in os.listdir(image_folder)
                  if model_name in img and
                  img.endswith("ms.png")]
        reordered_images = [(int(img.split(f" ")[0].split("_")[-1]), img) for img in images]
        reordered_images.sort()
    else:
        images = [img for img in os.listdir(image_folder)
                  if model_name in img and "layer" in img and
                  img.endswith("png")]
        reordered_images = [(int(img.split(f"layer_")[1].split("_")[0]), img) for img in images]
        reordered_images.sort()
    print(images)

    images = [elt[1] for elt in reordered_images]
    # Array images should only consider
    # the image files ignoring others if any
    print(images)

    frame = cv2.imread(os.path.join(image_folder, images[0]))

    # setting the frame width, height width
    # the width, height of first image
    height, width, layers = frame.shape
    print(height, width)

    video = cv2.VideoWriter(video_name, 0, 1, (width, height))

    # Appending the images to the video one by one
    for i, image in enumerate(images):
        img = cv2.imread(os.path.join(image_folder, image))
        BLACK = (255, 0, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 1.1
        font_color = BLACK
        font_thickness = 2
        text = ''
        x, y = 10, 650
        img_text = cv2.putText(img, text, (x, y), font, font_size, font_color, font_thickness, cv2.LINE_AA)

        video.write(img_text)

    # Deallocating memories taken for window creation
    cv2.destroyAllWindows()
    video.release()  # releasing the video generated
    return video_name
