import cv2
import torch
import torchvision.transforms as T
import glob 

from repnet.model import RepNet

import time
# start_time = time.time() 
# video_path = './event_video/class2/user02_lab.mp4' # './videos/cheetah_running_at_63_mph_102_kph.mp4' # path to video file

def inference(model, class_name, user_name):
    video_path = f'./event_video/{class_name}/{user_name}' # './videos/cheetah_running_at_63_mph_102_kph.mp4' # path to video file

    # repnet model variables 
    weights = './pytorch_weights.pth'
    device = 'cuda'
    strides = [1, 2, 3, 4, 8]
    fps = 60 

    # read frames and apply preprocessing 
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((112, 112)),
        T.ToTensor(),
        T.Normalize(mean=0.5, std=0.5),      
    ])

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    raw_frames, frames = [], []

    # frames_count = 600
    while cap.isOpened() :
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        raw_frames.append(frame)
        frame = transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frames.append(frame)
        # if not frames_count:
        #     break
        # frames_count -= 1
    cap.release()

    # Test multiple strides and pick the best one
    # print('Running inference on multiple stride values...')
    best_stride, best_confidence, best_period_length, best_period_count, best_periodicity_score, best_embeddings = None, None, None, None, None, None
    for stride in strides:
        # Apply stride
        stride_frames = frames[::stride]
        stride_frames = stride_frames[:(len(stride_frames) // 64) * 64]
        if len(stride_frames) < 64:
            continue # Skip this stride if there are not enough frames
        stride_frames = torch.stack(stride_frames, axis=0).unflatten(0, (-1, 64)).movedim(1, 2) # Convert to N x C x D x H x W
        stride_frames = stride_frames.to(device)
        # Run inference
        raw_period_length, raw_periodicity_score, embeddings = [], [], []
        with torch.no_grad():
            for i in range(stride_frames.shape[0]):  # Process each batch separately to avoid OOM
                batch_period_length, batch_periodicity, batch_embeddings = model(stride_frames[i].unsqueeze(0))
                raw_period_length.append(batch_period_length[0].cpu())
                raw_periodicity_score.append(batch_periodicity[0].cpu())
                embeddings.append(batch_embeddings[0].cpu())
        # Post-process results
        raw_period_length, raw_periodicity_score, embeddings = torch.cat(raw_period_length), torch.cat(raw_periodicity_score), torch.cat(embeddings)
        confidence, period_length, period_count, periodicity_score = model.get_counts(raw_period_length, raw_periodicity_score, stride)
        if best_confidence is None or confidence > best_confidence:
            best_stride, best_confidence, best_period_length, best_period_count, best_periodicity_score, best_embeddings = stride, confidence, period_length, period_count, periodicity_score, embeddings
    if best_stride is None:
        raise RuntimeError('The stride values used are too large and nove 64 video chunk could be sampled. Try different values for --strides.')
    # print(f'Predicted a period length of {best_period_length/fps:.1f} seconds (~{int(best_period_length)} frames) with a confidence of {best_confidence:.2f} using a stride of {best_stride} frames.')
    # print(f'Predicted a period count is {round(best_period_count.tolist()[-1])}')

    print(f"{class_name}, {user_name}, {round(period_count.tolist()[-1])}")

    # print("Time taken: ", time.time() - start_time)



if __name__ == "__main__":
    weights = './pytorch_weights.pth'
    device = 'cuda'
    # Load model
    model = RepNet() 
    state_dict = torch.load(weights)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    files = sorted(glob.glob('./event_video/class?/*.mp4'))
    for file in files:
        class_name = file.split('/')[-2]
        user_name = file.split('/')[-1]
        # print(f"Class name: {class_name}, User name: {user_name}")
        inference(model, class_name, user_name)