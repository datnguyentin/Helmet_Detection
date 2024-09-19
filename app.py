import streamlit as st
import cv2
import tempfile

import model_function as mf
import run
import cv2



def main_display():
    frame_placeholder = st.empty()
    black_placeholder = cv2.imread('asset/ui_bg/black_placeholder.jpg')
    frame_placeholder.image(black_placeholder)
    left_col, mid_col, right_col = st.columns(3)
    display_opt = left_col.selectbox("Display Method", ['opencv (Faster Display)', 'In Web (More Compact)'])
    run_opt = mid_col.selectbox("Run Method", ['Continuous', 'Frame-by-Frame(Only available in opencv display)'])
    process_button = right_col.button("Process")

    return frame_placeholder, display_opt, process_button, run_opt

def opt_video_local():
    frame_placeholder, display_opt, process_button, run_option = main_display()
    
    video = st.file_uploader("Upload Your video here", type = ['MP4', 'MOV'])
    frame_placeholder.video(video)
    if process_button:
        if video == None:
            st.text("No video found")
    
        else:
            #Convert to cv2 readable format
            tfile = tempfile.NamedTemporaryFile(delete = False)
            tfile.write(video.read())
            if display_opt == "In Web (More Compact)":
                run.process(tfile.name, app_running= True, frame_placeholder = frame_placeholder)
            else:
                run.process(tfile.name, app_running = True, run_option = run_option)


def opt_video_website(link):
    frame_placeholder, display_opt, process_button, run_option = main_display()

    if link:
        frame_placeholder.video(link)
    if process_button:
        if not link.startswith('http'):
            st.text("No video found")
    
        else:
            #Convert to cv2 readable format
            if display_opt == "In Web (More Compact)":
                run.process(link, app_running= True, frame_placeholder = frame_placeholder)
            else:
                run.process(link, app_running = True, run_option = run_option)



def opt_webcam():
    frame_placeholder, display_opt, process_button, run_option = main_display()

    if process_button:
        if display_opt == "In Web (More Compact)":
            run.process(0, app_running= True, frame_placeholder = frame_placeholder)
        else:
            if run_option == "Frame-by-Frame(Only available in opencv display)":
                st.text("Not available in Webcam mode, auto use 'Continuous'")
            run.process(0, app_running = True, run_option = "Continuous")

    
     
def side_bar():
    st.markdown(
        """
        <style>
        [data-testid = 'stSidebar'][aria-expanded = 'true'] > div:first-child{
            width: 350px
        }
        [data-testid = 'stSidebar'][aria-expanded = 'false'] > div:first-child{
            width: 350px
            margin-left: -350x
        }
        </style>
        """,
        unsafe_allow_html = True
    )

    st.sidebar.title("Violated Images")
    


def ui():
    st.title("Helmet Detection")
    side_bar()
    app_mode = st.selectbox("Video Source", ['video(local)', 'video(website)', 'webcam'])

    if app_mode == "video(local)":
        opt_video_local()
    elif app_mode == "video(website)":
        link = st.text_input("Provide link to video here")   
        opt_video_website(link)
    else:
        opt_webcam()
    
    

    

if __name__ == "__main__":

    #Initialize models
    helmet_detector, helmet_label_map = mf.load_helmet_detector()
    plate_ocr, char_label_map = mf.load_plate_ocr()
    tracker = mf.load_tracker()

    #Run UI
    ui()