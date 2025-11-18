import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
import enchant
import os
import math
import tkinter as tk
from PIL import Image, ImageTk
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector

# --- Configuration ---
# Paths to your trained models
# IMPORTANT: Update these paths to where your .h5 model files are located
CNN_MODEL_PATH = './cnn8grps_rad1_model.h5'
LSTM_MODEL_PATH = './final_lstm5.h5' # From main4.py

# Global parameters from final_pred.py
OFFSET = 29 # Offset for hand cropping
CNN_IMG_SIZE = (400, 400) # Target image size for CNN input

# LSTM Model specific parameters from main4.py and data_collection.py
LSTM_ACTIONS = np.array(['_', 'Hello', 'Thank You','Yes','No','I am Fine','Nice To meet You','good','Good Morning'])

LSTM_SEQUENCE_LENGTH = 60 # Number of frames to collect for one LSTM prediction

# Character labels for CNN (assuming 0=A, 1=B, etc., based on final_pred.py's logic)
# This mapping needs to align with how your cnn8grps_rad1_model.h5 was trained
# and how the custom logic in predict_cnn_character refines the predictions.
# This map provides the initial character group based on the CNN's raw output.
CNN_CHAR_GROUP_MAP = {
    0: 'S', # Primary character for group 0, refined later to A, T, E, M, N
    1: 'B', # Primary character for group 1, refined later to D, F, I, W, K, U, V, R
    2: 'C', # Primary character for group 2, refined later to O
    3: 'G', # Primary character for group 3, refined later to H
    4: 'L',
    5: 'P', # Primary character for group 5, refined later to Q, Z
    6: 'X',
    7: 'Y'  # Primary character for group 7, refined later to J
}

class Application:
    def __init__(self):
        # --- Initialize Tkinter Window ---
        self.root = tk.Tk()
        self.root.title("Sign Language To Text Conversion (Combined)")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("1400x950") # Increased height to ensure buttons are visible

        # --- Video Capture ---
        self.vs = cv2.VideoCapture(0)
        if not self.vs.isOpened():
            print("❌ Cannot access camera. Please check if it's in use or connected.")
            self.root.destroy()
            exit()

        self.current_image = None
        # Initialize with a blank image to avoid cv2.cvtColor errors on first frame
        self.current_cnn_input_image = np.ones((CNN_IMG_SIZE[0], CNN_IMG_SIZE[1], 3), np.uint8) * 255 

        # --- Load Models ---
        try:
            self.cnn_model = load_model(CNN_MODEL_PATH)
            print(f"CNN Model loaded from {CNN_MODEL_PATH}")
        except Exception as e:
            print(f"Error loading CNN model: {e}. Please ensure the path is correct.")
            self.root.destroy()
            exit()

        try:
            self.lstm_model = load_model(LSTM_MODEL_PATH)
            print(f"LSTM Model loaded from {LSTM_MODEL_PATH}")
        except Exception as e:
            print(f"Error loading LSTM model: {e}. Please ensure the path is correct.")
            self.root.destroy()
            exit()

        self.ddd = enchant.Dict("en-US") # For CNN word suggestions

        # --- Hand Detectors ---
        self.hd = HandDetector(maxHands=1) # For CNN input frame processing
        self.hd2 = HandDetector(maxHands=1) # For CNN hand segment processing

        # --- Mediapipe for LSTM ---
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils # This is now an instance variable
        self.holistic_detector = self.mp_holistic.Holistic(
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75
        )
        self.lstm_keypoints_sequence = [] # Sequence buffer for LSTM

        # --- Text-to-Speech Engine ---
        self.speak_engine = pyttsx3.init()
        self.speak_engine.setProperty("rate", 100)
        voices = self.speak_engine.getProperty("voices")
        if voices:
            self.speak_engine.setProperty("voice", voices[0].id)
        else:
            print("No speech synthesis voices found.")

        # --- Prediction Buffers and Variables ---
        self.current_sentence = [] # Stores recognized words/characters for display (CNN and LSTM combined)
        self.cnn_current_word_buffer = "" # For building words from CNN predictions before adding to sentence

        # CNN-specific debouncing/smoothing variables (from final_pred.py)
        # These mimic the logic found directly in final_pred.py's predict method
        self.prev_char = " " # Character that was last *successfully added* to self.str
        self.count = -1 # Frame counter, increments every frame in video_loop
        self.ten_prev_char = [" "] * 10 # Circular buffer for last 10 raw CNN predictions (after internal refinement)

        self.last_lstm_word_prediction = "" # To avoid repeating the same word from LSTM
        self.current_symbol = " " # Current immediate CNN character prediction display (after refinement)
        self.current_lstm_word = " " # Current immediate LSTM word prediction display

        # Word suggestions from enchant (cnn_pred)
        self.word1 = " "
        self.word2 = " "
        self.word3 = " "
        self.word4 = " "

        # Prediction Mode variable
        # 'both': LSTM and CNN active (LSTM prioritized for words)
        # 'phrase': Only LSTM active
        # 'letter': Only CNN active
        self.prediction_mode = 'both' # Default mode

        # Create a blank white image to use as background for CNN input visualization
        self.white_bg_for_cnn_drawing = np.ones((CNN_IMG_SIZE[0], CNN_IMG_SIZE[1], 3), np.uint8) * 255

        # --- GUI Elements ---
        self.T = tk.Label(self.root, text="Sign Language To Text Conversion (Combined)", font=("Courier", 30, "bold"))
        self.T.place(x=10, y=5)

        # Status Label for user feedback
        self.status_label = tk.Label(self.root, text="Application Ready", font=("Courier", 15), fg="green")
        self.status_label.place(x=10, y=40)

        # Main video panel (webcam feed with Mediapipe landmarks)
        self.panel_main_video = tk.Label(self.root)
        self.panel_main_video.place(x=10, y=80, width=640, height=480) # Larger video feed

        # CNN input visualization panel (hand drawing on white background)
        self.panel_cnn_input = tk.Label(self.root)
        self.panel_cnn_input.place(x=700, y=80, width=CNN_IMG_SIZE[0], height=CNN_IMG_SIZE[1]) # Display 400x400 CNN input

        # Current Symbol/Character display
        self.T1 = tk.Label(self.root, text="CNN Char:", font=("Courier", 20, "bold"))
        self.T1.place(x=10, y=570)
        self.panel_current_char = tk.Label(self.root, text=self.current_symbol, font=("Courier", 20))
        self.panel_current_char.place(x=180, y=570)

        # Current LSTM Word display
        self.T_lstm = tk.Label(self.root, text="LSTM Word:", font=("Courier", 20, "bold"))
        self.T_lstm.place(x=10, y=610)
        self.panel_current_lstm_word = tk.Label(self.root, text=self.current_lstm_word, font=("Courier", 20))
        self.panel_current_lstm_word.place(x=180, y=610)

        # Sentence display
        self.T3 = tk.Label(self.root, text="Sentence:", font=("Courier", 25, "bold"))
        self.T3.place(x=10, y=650)
        self.panel_sentence = tk.Label(self.root, text=" ", font=("Courier", 25), wraplength=1000, justify=tk.LEFT, anchor='nw')
        self.panel_sentence.place(x=180, y=650, width=1200, height=80)

        # Suggestions label
        self.T4 = tk.Label(self.root, text="Suggestions:", fg="red", font=("Courier", 20, "bold"))
        self.T4.place(x=700, y=570)

        # Suggestion buttons
        self.b1 = tk.Button(self.root, text=" ", font=("Courier", 15), command=lambda: self.action_suggest(self.word1))
        self.b1.place(x=700, y=600, width=150, height=40)
        self.b2 = tk.Button(self.root, text=" ", font=("Courier", 15), command=lambda: self.action_suggest(self.word2))
        self.b2.place(x=860, y=600, width=150, height=40)
        self.b3 = tk.Button(self.root, text=" ", font=("Courier", 15), command=lambda: self.action_suggest(self.word3))
        self.b3.place(x=1020, y=600, width=150, height=40)
        self.b4 = tk.Button(self.root, text=" ", font=("Courier", 15), command=lambda: self.action_suggest(self.word4))
        self.b4.place(x=1180, y=600, width=150, height=40)

        # Action Buttons (adjusted Y-coordinate for visibility)
        self.speak_btn = tk.Button(self.root, text="Speak", font=("Courier", 20), command=self.speak_fun)
        self.speak_btn.place(x=10, y=820, width=150, height=50)

        self.clear_btn = tk.Button(self.root, text="Clear", font=("Courier", 20), command=self.clear_fun)
        self.clear_btn.place(x=170, y=820, width=150, height=50)

        # Mode Switching Buttons (adjusted Y-coordinate for visibility)
        self.btn_phrase_mode = tk.Button(self.root, text="Phrase Mode (LSTM)", font=("Courier", 15), command=self.set_phrase_mode)
        self.btn_phrase_mode.place(x=340, y=820, width=200, height=50)

        self.btn_letter_mode = tk.Button(self.root, text="Letter Mode (CNN)", font=("Courier", 15), command=self.set_letter_mode)
        self.btn_letter_mode.place(x=550, y=820, width=200, height=50)

        self.btn_both_modes = tk.Button(self.root, text="Both Modes", font=("Courier", 15), command=self.set_both_modes)
        self.btn_both_modes.place(x=760, y=820, width=200, height=50)
        
        self.panel_mode = tk.Label(self.root, text=f"Mode: {self.prediction_mode.upper()}", font=("Courier", 15, "bold"), fg="blue")
        self.panel_mode.place(x=1000, y=835)

        # --- Keyboard Bindings for Mode Switching ---
        self.root.bind('<KeyPress-p>', lambda event: self.set_phrase_mode())
        self.root.bind('<KeyPress-l>', lambda event: self.set_letter_mode())
        self.root.bind('<KeyPress-b>', lambda event: self.set_both_modes())


        # --- Start Video Loop ---
        self.video_loop()

    # Moved helper function into the class as a method
    def draw_holistic_landmarks(self, image, results):
        """
        Draw hand landmarks on the image.
        Args:
            image (numpy.ndarray): The input image.
            results: The landmarks detected by Mediapipe.
        Returns:
            image (numpy.ndarray): The image with landmarks drawn.
        """
        output_image = image.copy()
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks( # Now uses self.mp_drawing
                output_image,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                self.mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
            )
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks( # Now uses self.mp_drawing
                output_image,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )
        return output_image

    def video_loop(self):
        try:
            ok, frame = self.vs.read()
            if not ok:
                self.root.after(10, self.video_loop) # Try again if frame not read
                return

            frame = cv2.flip(frame, 1) # Flip for selfie view
            display_frame = frame.copy() # Frame to overlay all info

            # --- LSTM Stream Processing (Active in 'phrase' or 'both' modes) ---
            self.current_lstm_word = " " # Reset for current frame if no prediction this frame
            if self.prediction_mode in ['phrase', 'both']:
                results_holistic = image_process_holistic(display_frame, self.holistic_detector)
                # Draw holistic landmarks on the main frame for visualization
                display_frame = self.draw_holistic_landmarks(display_frame, results_holistic) # Call as method

                if results_holistic.left_hand_landmarks or results_holistic.right_hand_landmarks:
                    lstm_keypoints = keypoint_extraction_holistic(results_holistic)
                    # Only append if keypoints are not all zeros (i.e., hand was detected)
                    if not np.all(lstm_keypoints == 0):
                        self.lstm_keypoints_sequence.append(lstm_keypoints)

                    if len(self.lstm_keypoints_sequence) == LSTM_SEQUENCE_LENGTH:
                        sequence_np = np.expand_dims(np.array(self.lstm_keypoints_sequence), axis=0).astype('float32')
                        if sequence_np.size > 0: # Ensure array is not empty
                            lstm_prediction_probs = self.lstm_model.predict(sequence_np)[0]
                            predicted_action_index = np.argmax(lstm_prediction_probs)

                            if np.amax(lstm_prediction_probs) > 0.9: # Confidence threshold for LSTM
                                predicted_word = LSTM_ACTIONS[predicted_action_index]
                                if predicted_word != 'None' and predicted_word != self.last_lstm_word_prediction:
                                    # When LSTM predicts a word, clear CNN buffer and add word to sentence
                                    if self.cnn_current_word_buffer.strip():
                                        self.current_sentence.append(self.cnn_current_word_buffer.strip())
                                        self.cnn_current_word_buffer = ""
                                    
                                    # Add space if sentence isn't empty and doesn't end with space
                                    if self.current_sentence and self.current_sentence[-1] != " " and predicted_word != "_":
                                        self.current_sentence.append(" ")
                                    
                                    # Only add if it's not a placeholder
                                    if predicted_word != "_": # Assuming "_" is a placeholder for 'None' or blank from LSTM_ACTIONS
                                        self.current_sentence.append(predicted_word)
                                    
                                    self.last_lstm_word_prediction = predicted_word
                                self.current_lstm_word = predicted_word # Update current LSTM word display
                            else:
                                self.last_lstm_word_prediction = "" # Reset if confidence drops or 'None'
                        self.lstm_keypoints_sequence = [] # Reset sequence after prediction

            # --- CNN Stream Processing (Active in 'letter' or 'both' modes) ---
            self.current_cnn_input_image = self.white_bg_for_cnn_drawing.copy() # Start with fresh bg
            
            raw_cnn_char_prediction = " " # Initialize for this frame's raw prediction

            if self.prediction_mode in ['letter', 'both']:
                hands_list_cvzone, _ = self.hd.findHands(frame, draw=False, flipType=True)
                hand_detected_for_cnn = bool(hands_list_cvzone) # Flag to pass to predict_cnn_character

                if hand_detected_for_cnn:
                    hand_cvzone_dict = hands_list_cvzone[0]
                    x, y, w, h = hand_cvzone_dict['bbox']

                    img_segment_for_cnn = frame[max(0, y - OFFSET):min(frame.shape[0], y + h + OFFSET),
                                                max(0, x - OFFSET):min(frame.shape[1], x + w + OFFSET)]
                    
                    if img_segment_for_cnn.shape[0] > 0 and img_segment_for_cnn.shape[1] > 0:
                        handz_list_cvzone_in_segment, _ = self.hd2.findHands(img_segment_for_cnn, draw=False, flipType=True)
                        
                        if handz_list_cvzone_in_segment:
                            hand_in_segment_dict = handz_list_cvzone_in_segment[0]
                            self.pts = hand_in_segment_dict['lmList'] # Store pts for predict_cnn_character
                            
                            # Draw landmarks on white background for CNN input (replicated from final_pred.py)
                            os_x = ((CNN_IMG_SIZE[0] - w) // 2) - 15
                            os_y = ((CNN_IMG_SIZE[1] - h) // 2) - 15

                            for t in range(0, 4, 1): cv2.line(self.current_cnn_input_image, (self.pts[t][0] + os_x, self.pts[t][1] + os_y), (self.pts[t + 1][0] + os_x, self.pts[t + 1][1] + os_y), (0, 255, 0), 3)
                            for t in range(5, 8, 1): cv2.line(self.current_cnn_input_image, (self.pts[t][0] + os_x, self.pts[t][1] + os_y), (self.pts[t + 1][0] + os_x, self.pts[t + 1][1] + os_y), (0, 255, 0), 3)
                            for t in range(9, 12, 1): cv2.line(self.current_cnn_input_image, (self.pts[t][0] + os_x, self.pts[t][1] + os_y), (self.pts[t + 1][0] + os_x, self.pts[t + 1][1] + os_y), (0, 255, 0), 3)
                            for t in range(13, 16, 1): cv2.line(self.current_cnn_input_image, (self.pts[t][0] + os_x, self.pts[t][1] + os_y), (self.pts[t + 1][0] + os_x, self.pts[t + 1][1] + os_y), (0, 255, 0), 3)
                            for t in range(17, 20, 1): cv2.line(self.current_cnn_input_image, (self.pts[t][0] + os_x, self.pts[t][1] + os_y), (self.pts[t + 1][0] + os_x, self.pts[t + 1][1] + os_y), (0, 255, 0), 3)
                            cv2.line(self.current_cnn_input_image, (self.pts[5][0] + os_x, self.pts[5][1] + os_y), (self.pts[9][0] + os_x, self.pts[9][1] + os_y), (0, 255, 0), 3)
                            cv2.line(self.current_cnn_input_image, (self.pts[9][0] + os_x, self.pts[9][1] + os_y), (self.pts[13][0] + os_x, self.pts[13][1] + os_y), (0, 255, 0), 3)
                            cv2.line(self.current_cnn_input_image, (self.pts[13][0] + os_x, self.pts[13][1] + os_y), (self.pts[17][0] + os_x, self.pts[17][1] + os_y), (0, 255, 0), 3)
                            cv2.line(self.current_cnn_input_image, (self.pts[0][0] + os_x, self.pts[0][1] + os_y), (self.pts[5][0] + os_x, self.pts[5][1] + os_y), (0, 255, 0), 3)
                            cv2.line(self.current_cnn_input_image, (self.pts[0][0] + os_x, self.pts[0][1] + os_y), (self.pts[17][0] + os_x, self.pts[17][1] + os_y), (0, 255, 0), 3)
                            for i in range(21): cv2.circle(self.current_cnn_input_image, (self.pts[i][0] + os_x, self.pts[i][1] + os_y), 2, (0, 0, 255), 1)

                            raw_cnn_char_prediction = self.predict_cnn_character(self.current_cnn_input_image, self.pts, hand_detected_for_cnn) # Pass self.pts
                        else:
                            raw_cnn_char_prediction = " " # Default to blank if no hand landmarks in segment
                    else:
                        raw_cnn_char_prediction = " " # Default to blank if segment is invalid
                else:
                    raw_cnn_char_prediction = " " # Default to blank if no hand detected

                self.current_symbol = raw_cnn_char_prediction # Always display the current raw prediction (after refinement)

                # --- CNN Debouncing and Sentence Building Logic (DIRECTLY from final_pred.py's predict method) ---
                self.count += 1
                self.ten_prev_char[self.count % 10] = raw_cnn_char_prediction # Store current raw prediction

                # Logic from final_pred.py:
                # If current char is "next" (SPACE) and previous committed char was not "next"
                if raw_cnn_char_prediction == "SPACE" and self.prev_char != "SPACE":
                    # Check if the character 2 frames ago was also "SPACE" (consistency check)
                    if self.count >= 2 and self.ten_prev_char[(self.count - 2) % 10] == "SPACE":
                        # If there's a word in the buffer, append it to the sentence
                        if self.cnn_current_word_buffer.strip():
                            self.current_sentence.append(self.cnn_current_word_buffer.strip())
                            self.cnn_current_word_buffer = ""
                        # Add a space to the sentence if it's not already empty or ending with a space
                        if not self.current_sentence or (self.current_sentence and self.current_sentence[-1] != " "):
                            self.current_sentence.append(" ")
                        self.prev_char = "SPACE" # Mark SPACE as the last committed action
                
                # If current char is "BACKSPACE" and previous committed char was not "BACKSPACE"
                elif raw_cnn_char_prediction == "BACKSPACE" and self.prev_char != "BACKSPACE":
                    # Check if the character 2 frames ago was also "BACKSPACE"
                    if self.count >= 2 and self.ten_prev_char[(self.count - 2) % 10] == "BACKSPACE":
                        if self.cnn_current_word_buffer:
                            self.cnn_current_word_buffer = self.cnn_current_word_buffer[:-1] # Remove last char from buffer
                        elif self.current_sentence:
                            # Remove the last element from the sentence
                            if self.current_sentence:
                                last_element = self.current_sentence.pop()
                                # If the removed element was a space, and there's a word before it, remove that word too
                                if last_element == " " and self.current_sentence:
                                    self.current_sentence.pop()
                        self.prev_char = "BACKSPACE" # Mark BACKSPACE as the last committed action

                # For regular characters (not SPACE/BACKSPACE/blank) and if different from prev_char
                elif raw_cnn_char_prediction not in ["SPACE", "BACKSPACE", " "] and raw_cnn_char_prediction != self.prev_char:
                    # Check for consistency of the current character over the last few frames (3 frames)
                    if self.count >= 2 and \
                       self.ten_prev_char[(self.count - 2) % 10] == raw_cnn_char_prediction and \
                       self.ten_prev_char[(self.count - 1) % 10] == raw_cnn_char_prediction:
                        
                        if self.prev_char == "SPACE": # If the last committed action was a space, start a new word
                            self.cnn_current_word_buffer = "" 
                        self.cnn_current_word_buffer += raw_cnn_char_prediction # Append to current word buffer
                        self.prev_char = raw_cnn_char_prediction # Mark current char as last committed

                # If the current prediction is blank (" "), and the previously committed character was not blank,
                # then reset prev_char to " " to allow a new character to be committed after this blank sequence.
                # This ensures that if the hand disappears and reappears, a new char can be registered.
                if raw_cnn_char_prediction == " " and self.prev_char != " ":
                    self.prev_char = " "

                # Get word suggestions using enchant for the current CNN buffer
                current_text_for_suggestions = "".join(self.current_sentence) + self.cnn_current_word_buffer
                if current_text_for_suggestions.strip():
                    st = current_text_for_suggestions.rfind(" ")
                    ed = len(current_text_for_suggestions)
                    word = current_text_for_suggestions[st+1:ed] # Extract the last word for suggestions
                    
                    if len(word.strip())!=0:
                        self.ddd.check(word) # This line checks if the word is in the dictionary
                        suggestions = self.ddd.suggest(word)
                        # Assign suggestions to buttons, padding with blanks if fewer than 4
                        self.word1 = suggestions[0].upper() if len(suggestions) >= 1 else " "
                        self.word2 = suggestions[1].upper() if len(suggestions) >= 2 else " "
                        self.word3 = suggestions[2].upper() if len(suggestions) >= 3 else " "
                        self.word4 = suggestions[3].upper() if len(suggestions) >= 4 else " "
                    else: # Clear suggestions if no meaningful word
                        self.word1 = self.word2 = self.word3 = self.word4 = " " 
                else: # Clear all suggestions if overall text is blank
                    self.word1 = self.word2 = self.word3 = self.word4 = " "

        except Exception as e:
            print(f"Error in video_loop: {e}")
            # print(traceback.format_exc()) # Uncomment for detailed traceback

        # --- Update GUI ---
        # Update main video panel
        img_main = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
        img_main = img_main.resize((640, 480)) # Resize to panel size
        imgtk_main = ImageTk.PhotoImage(image=img_main)
        self.panel_main_video.imgtk = imgtk_main
        self.panel_main_video.config(image=imgtk_main)

        # Update CNN input panel
        # Robust check before converting image to prevent _src.empty() error
        if self.current_cnn_input_image is None or self.current_cnn_input_image.size == 0 or self.current_cnn_input_image.shape[0] == 0 or self.current_cnn_input_image.shape[1] == 0:
            self.current_cnn_input_image = self.white_bg_for_cnn_drawing.copy() # Ensure it's a valid image

        img_cnn = Image.fromarray(cv2.cvtColor(self.current_cnn_input_image, cv2.COLOR_BGR2RGB))
        imgtk_cnn = ImageTk.PhotoImage(image=img_cnn)
        self.panel_cnn_input.imgtk = imgtk_cnn
        self.panel_cnn_input.config(image=imgtk_cnn)

        # Update text labels and buttons
        self.panel_current_char.config(text=self.current_symbol)
        self.panel_current_lstm_word.config(text=self.current_lstm_word)

        # Combine sentence parts for final display
        final_display_text = "".join(self.current_sentence)
        if self.cnn_current_word_buffer.strip():
            # If current sentence ends with space and buffer is starting with a char, don't add another space
            if final_display_text and final_display_text[-1] == " ":
                final_display_text += self.cnn_current_word_buffer
            else:
                final_display_text += " " + self.cnn_current_word_buffer if final_display_text else self.cnn_current_word_buffer

        self.panel_sentence.config(text=final_display_text.strip()) # .strip() to remove leading/trailing spaces

        self.b1.config(text=self.word1)
        self.b2.config(text=self.word2)
        self.b3.config(text=self.word3)
        self.b4.config(text=self.word4)
        
        self.panel_mode.config(text=f"Mode: {self.prediction_mode.upper()}")


        # Schedule next loop
        self.root.after(10, self.video_loop)

    def euclidean_distance(self, p1, p2):
        """Calculates Euclidean distance between two points."""
        return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))

    def predict_cnn_character(self, cnn_img_input, pts_cvzone, hand_detected: bool):
        """
        Predicts a character using the CNN model and applies custom logic from final_pred.py.
        This function now returns the *raw* character prediction after applying
        the conditional rules, but without debouncing or adding to the sentence.
        Args:
            cnn_img_input (numpy.ndarray): The 400x400x3 image prepared for CNN.
            pts_cvzone (list): List of 21 (x,y) landmark points from cvzone.
            hand_detected (bool): True if a hand was detected in the frame, False otherwise.
        Returns:
            str: Raw predicted character (after subgroup refinement, before debouncing).
        """
        if not hand_detected or pts_cvzone is None or not pts_cvzone:
            return " " # Return blank if no hand or no landmarks

        cnn_input_processed = cnn_img_input.reshape(1, CNN_IMG_SIZE[0], CNN_IMG_SIZE[1], 3).astype('float32') / 255.0

        prob = np.array(self.cnn_model.predict(cnn_input_processed)[0], dtype='float32')
        ch1_idx = np.argmax(prob, axis=0)
        prob[ch1_idx] = 0
        ch2_idx = np.argmax(prob, axis=0)
        # Note: final_pred.py also extracts ch3 but doesn't seem to use it in the logic below.
        # Keeping consistent by not using it here either.

        pl = [ch1_idx, ch2_idx] # Pair of top 2 predictions

        # --- Apply custom conditional logic from final_pred.py ---
        # These conditions refine the initial CNN prediction based on landmark positions
        # and the top two predicted groups.

        # condition for [Aemnst] (Group 0)
        l_cond0 = [[5, 2], [5, 3], [3, 5], [3, 6], [3, 0], [3, 2], [6, 4], [6, 1], [6, 2], [6, 6], [6, 7], [6, 0], [6, 5],
                 [4, 1], [1, 0], [1, 1], [6, 3], [1, 6], [5, 6], [5, 1], [4, 5], [1, 4], [1, 5], [2, 0], [2, 6], [4, 6],
                 [1, 0], [5, 7], [1, 6], [6, 1], [7, 6], [2, 5], [7, 1], [5, 4], [7, 0], [7, 5], [7, 2]]
        if pl in l_cond0:
            if (pts_cvzone[6][1] < pts_cvzone[8][1] and pts_cvzone[10][1] < pts_cvzone[12][1] and
                pts_cvzone[14][1] < pts_cvzone[16][1] and pts_cvzone[18][1] < pts_cvzone[20][1]):
                ch1_idx = 0

        # condition for [o][s]
        l_cond_os = [[2, 2], [2, 1]]
        if pl in l_cond_os:
            if (pts_cvzone[5][0] < pts_cvzone[4][0]):
                ch1_idx = 0

        # condition for [c0][aemnst]
        l_cond_c0_aemnst1 = [[0, 0], [0, 6], [0, 2], [0, 5], [0, 1], [0, 7], [5, 2], [7, 6], [7, 1]]
        if pl in l_cond_c0_aemnst1:
            if (pts_cvzone[0][0] > pts_cvzone[8][0] and pts_cvzone[0][0] > pts_cvzone[4][0] and
                pts_cvzone[0][0] > pts_cvzone[12][0] and pts_cvzone[0][0] > pts_cvzone[16][0] and
                pts_cvzone[0][0] > pts_cvzone[20][0]) and pts_cvzone[5][0] > pts_cvzone[4][0]:
                ch1_idx = 2

        # condition for [c0][aemnst] (another rule)
        l_cond_c0_aemnst2 = [[6, 0], [6, 6], [6, 2]]
        if pl in l_cond_c0_aemnst2:
            if self.euclidean_distance(pts_cvzone[8], pts_cvzone[16]) < 52:
                ch1_idx = 2


        # condition for [gh][bdfikruvw]
        l_cond_gh_bdf = [[1, 4], [1, 5], [1, 6], [1, 3], [1, 0]]
        if pl in l_cond_gh_bdf:
            if pts_cvzone[6][1] > pts_cvzone[8][1] and pts_cvzone[14][1] < pts_cvzone[16][1] and pts_cvzone[18][1] < pts_cvzone[20][1] and \
               pts_cvzone[0][0] < pts_cvzone[8][0] and pts_cvzone[0][0] < pts_cvzone[12][0] and pts_cvzone[0][0] < pts_cvzone[16][0] and pts_cvzone[0][0] < pts_cvzone[20][0]:
                ch1_idx = 3



        # con for [gh][l]
        l_cond_gh_l = [[4, 6], [4, 1], [4, 5], [4, 3], [4, 7]]
        if pl in l_cond_gh_l:
            if (pts_cvzone[4][0] > pts_cvzone[0][0]):
                ch1_idx = 3

        # con for [gh][pqz]
        l_cond_gh_pqz = [[5, 3], [5, 0], [5, 7], [5, 4], [5, 2], [5, 1], [5, 5]]
        if pl in l_cond_gh_pqz:
            if (pts_cvzone[2][1] + 15 < pts_cvzone[16][1]):
                ch1_idx = 3

        # con for [l][x]
        l_cond_l_x = [[6, 4], [6, 1], [6, 2]]
        if pl in l_cond_l_x:
            if self.euclidean_distance(pts_cvzone[4], pts_cvzone[11]) > 55:
                ch1_idx = 4

        # con for [l][d]
        l_cond_l_d = [[1, 4], [1, 6], [1, 1]]
        if pl in l_cond_l_d:
            if (self.euclidean_distance(pts_cvzone[4], pts_cvzone[11]) > 50) and \
               (pts_cvzone[6][1] > pts_cvzone[8][1] and pts_cvzone[10][1] < pts_cvzone[12][1] and pts_cvzone[14][1] < pts_cvzone[16][1] and pts_cvzone[18][1] < pts_cvzone[20][1]):
                ch1_idx = 4

        # con for [l][gh]
        l_cond_l_gh = [[3, 6], [3, 4]]
        if pl in l_cond_l_gh:
            if (pts_cvzone[4][0] < pts_cvzone[0][0]):
                ch1_idx = 4

        # con for [l][c0]
        l_cond_l_c0 = [[2, 2], [2, 5], [2, 4]]
        if pl in l_cond_l_c0:
            if (pts_cvzone[1][0] < pts_cvzone[12][0]):
                ch1_idx = 4
        
        # con for [gh][z] (repeated in final_pred.py)
        l_cond_gh_z = [[3, 6], [3, 5], [3, 4]]
        if pl in l_cond_gh_z:
            if (pts_cvzone[6][1] > pts_cvzone[8][1] and pts_cvzone[10][1] < pts_cvzone[12][1] and pts_cvzone[14][1] < pts_cvzone[16][1] and pts_cvzone[18][1] < pts_cvzone[20][1]) and pts_cvzone[4][1] > pts_cvzone[10][1]:
                ch1_idx = 5

        # con for [gh][pq] (repeated in final_pred.py)
        l_cond_gh_pq = [[3, 2], [3, 1], [3, 6]]
        if pl in l_cond_gh_pq:
            if pts_cvzone[4][1] + 17 > pts_cvzone[8][1] and pts_cvzone[4][1] + 17 > pts_cvzone[12][1] and pts_cvzone[4][1] + 17 > pts_cvzone[16][1] and pts_cvzone[4][1] + 17 > pts_cvzone[20][1]:
                ch1_idx = 5

        # con for [l][pqz] (repeated in final_pred.py)
        l_cond_l_pqz = [[4, 4], [4, 5], [4, 2], [7, 5], [7, 6], [7, 0]]
        if pl in l_cond_l_pqz:
            if pts_cvzone[4][0] > pts_cvzone[0][0]:
                ch1_idx = 5

        # con for [pqz][aemnst] (repeated in final_pred.py)
        l_cond_pqz_aemnst = [[0, 2], [0, 6], [0, 1], [0, 5], [0, 0], [0, 7], [0, 4], [0, 3], [2, 7]]
        if pl in l_cond_pqz_aemnst:
            if pts_cvzone[0][0] < pts_cvzone[8][0] and pts_cvzone[0][0] < pts_cvzone[12][0] and pts_cvzone[0][0] < pts_cvzone[16][0] and pts_cvzone[0][0] < pts_cvzone[20][0]:
                ch1_idx = 5

        # con for [pqz][yj] (repeated in final_pred.py)
        l_cond_pqz_yj = [[5, 7], [5, 2], [5, 6]]
        if pl in l_cond_pqz_yj:
            if pts_cvzone[3][0] < pts_cvzone[0][0]:
                ch1_idx = 7

        # con for [l][yj] (repeated in final_pred.py)
        l_cond_l_yj = [[4, 6], [4, 2], [4, 4], [4, 1], [4, 5], [4, 7]]
        if pl in l_cond_l_yj:
            if pts_cvzone[6][1] < pts_cvzone[8][1]:
                ch1_idx = 7

        # con for [x][yj] (repeated in final_pred.py)
        l_cond_x_yj = [[6, 7], [0, 7], [0, 1], [0, 0], [6, 4], [6, 6], [6, 5], [6, 1]]
        if pl in l_cond_x_yj:
            if pts_cvzone[18][1] > pts_cvzone[20][1]:
                ch1_idx = 7

        # condition for [x][aemnst] (repeated in final_pred.py)
        l_cond_x_aemnst = [[0, 4], [0, 2], [0, 3], [0, 1], [0, 6]]
        if pl in l_cond_x_aemnst:
            if pts_cvzone[5][0] > pts_cvzone[16][0]:
                ch1_idx = 6

        # condition for [yj][x] (repeated in final_pred.py)
        l_cond_yj_x = [[7, 2]]
        if pl in l_cond_yj_x:
            if pts_cvzone[18][1] < pts_cvzone[20][1] and pts_cvzone[8][1] < pts_cvzone[10][1]:
                ch1_idx = 6

        # condition for [c0][x] (repeated in final_pred.py)
        l_cond_c0_x = [[2, 1], [2, 2], [2, 6], [2, 7], [2, 0]]
        if pl in l_cond_c0_x:
            if self.euclidean_distance(pts_cvzone[8], pts_cvzone[16]) > 50:
                ch1_idx = 6

        # con for [l][x] (repeated in final_pred.py)
        l_cond_l_x2 = [[4, 6], [4, 2], [4, 1], [4, 4]]
        if pl in l_cond_l_x2:
            if self.euclidean_distance(pts_cvzone[4], pts_cvzone[11]) < 60:
                ch1_idx = 6

        # con for [x][d] (repeated in final_pred.py)
        l_cond_x_d = [[1, 4], [1, 6], [1, 0], [1, 2]]
        if pl in l_cond_x_d:
            if pts_cvzone[5][0] - pts_cvzone[4][0] - 15 > 0:
                ch1_idx = 6

        # con for [b][pqz] (repeated in final_pred.py)
        l_cond_b_pqz = [[5, 0], [5, 1], [5, 4], [5, 5], [5, 6], [6, 1], [7, 6], [0, 2], [7, 1], [7, 4], [6, 6], [7, 2], [5, 0],
                 [6, 3], [6, 4], [7, 5], [7, 2]]
        if pl in l_cond_b_pqz:
            if (pts_cvzone[6][1] > pts_cvzone[8][1] and pts_cvzone[10][1] > pts_cvzone[12][1] and pts_cvzone[14][1] > pts_cvzone[16][1] and pts_cvzone[18][1] > pts_cvzone[20][1]):
                ch1_idx = 1

        # con for [f][pqz] (repeated in final_pred.py)
        l_cond_f_pqz = [[6, 1], [6, 0], [0, 3], [6, 4], [2, 2], [0, 6], [6, 2], [7, 6], [4, 6], [4, 1], [4, 2], [0, 2], [7, 1],
             [7, 4], [6, 6], [7, 2], [7, 5], [7, 2]]
        if pl in l_cond_f_pqz:
            if (pts_cvzone[6][1] < pts_cvzone[8][1] and pts_cvzone[10][1] > pts_cvzone[12][1] and
                    pts_cvzone[14][1] > pts_cvzone[16][1] and pts_cvzone[18][1] > pts_cvzone[20][1]):
                ch1_idx = 1

        l_cond_f_pqz_alt = [[6, 1], [6, 0], [4, 2], [4, 1], [4, 6], [4, 4]]
        if pl in l_cond_f_pqz_alt:
            if (pts_cvzone[10][1] > pts_cvzone[12][1] and pts_cvzone[14][1] > pts_cvzone[16][1] and
                    pts_cvzone[18][1] > pts_cvzone[20][1]):
                ch1_idx = 1

        # con for [d][pqz] (repeated in final_pred.py)
        l_cond_d_pqz = [[5, 0], [3, 4], [3, 0], [3, 1], [3, 5], [5, 5], [5, 4], [5, 1], [7, 6]]
        if pl in l_cond_d_pqz:
            if ((pts_cvzone[6][1] > pts_cvzone[8][1] and pts_cvzone[10][1] < pts_cvzone[12][1] and pts_cvzone[14][1] < pts_cvzone[16][1] and
                 pts_cvzone[18][1] < pts_cvzone[20][1]) and (pts_cvzone[2][0] < pts_cvzone[0][0]) and pts_cvzone[4][1] > pts_cvzone[14][1]):
                ch1_idx = 1

        l_cond_d_pqz_alt1 = [[4, 1], [4, 2], [4, 4]]
        if pl in l_cond_d_pqz_alt1:
            if (self.euclidean_distance(pts_cvzone[4], pts_cvzone[11]) < 50) and (
                    pts_cvzone[6][1] > pts_cvzone[8][1] and pts_cvzone[10][1] < pts_cvzone[12][1] and pts_cvzone[14][1] < pts_cvzone[16][1] and pts_cvzone[18][1] <
                    pts_cvzone[20][1]):
                ch1_idx = 1

        l_cond_d_pqz_alt2 = [[3, 4], [3, 0], [3, 1], [3, 5], [3, 6]]
        if pl in l_cond_d_pqz_alt2:
            if ((pts_cvzone[6][1] > pts_cvzone[8][1] and pts_cvzone[10][1] < pts_cvzone[12][1] and pts_cvzone[14][1] < pts_cvzone[16][1] and
                 pts_cvzone[18][1] < pts_cvzone[20][1]) and (pts_cvzone[2][0] < pts_cvzone[0][0]) and pts_cvzone[14][1] < pts_cvzone[4][1]):
                ch1_idx = 1

        l_cond_d_pqz_alt3 = [[6, 6], [6, 4], [6, 1], [6, 2]]
        if pl in l_cond_d_pqz_alt3:
            if (pts_cvzone[5][0] - pts_cvzone[4][0] - 15 < 0):
                ch1_idx = 1

        # con for [i][pqz] (repeated in final_pred.py)
        l_cond_i_pqz = [[5, 4], [5, 5], [5, 1], [0, 3], [0, 7], [5, 0], [0, 2], [6, 2], [7, 5], [7, 1], [7, 6], [7, 7]]
        if pl in l_cond_i_pqz:
            if ((pts_cvzone[6][1] < pts_cvzone[8][1] and pts_cvzone[10][1] < pts_cvzone[12][1] and pts_cvzone[14][1] < pts_cvzone[16][1] and
                 pts_cvzone[18][1] > pts_cvzone[20][1])):
                ch1_idx = 1

        # con for [yj][bfdi] (repeated in final_pred.py)
        l_cond_yj_bfdi = [[1, 5], [1, 7], [1, 1], [1, 6], [1, 3], [1, 0]]
        if pl in l_cond_yj_bfdi:
            if (pts_cvzone[4][0] < pts_cvzone[5][0] + 15) and (
            (pts_cvzone[6][1] < pts_cvzone[8][1] and pts_cvzone[10][1] < pts_cvzone[12][1] and pts_cvzone[14][1] < pts_cvzone[16][1] and
             pts_cvzone[18][1] > pts_cvzone[20][1])):
                ch1_idx = 7

        # con for [uvr] (repeated in final_pred.py)
        l_cond_uvr = [[5, 5], [5, 0], [5, 4], [5, 1], [4, 6], [4, 1], [7, 6], [3, 0], [3, 5]]
        if pl in l_cond_uvr:
            if ((pts_cvzone[6][1] > pts_cvzone[8][1] and pts_cvzone[10][1] > pts_cvzone[12][1] and pts_cvzone[14][1] < pts_cvzone[16][1] and
                 pts_cvzone[18][1] < pts_cvzone[20][1])) and pts_cvzone[4][1] > pts_cvzone[14][1]:
                ch1_idx = 1

        # con for [w] (repeated in final_pred.py)
        l_cond_w1 = [[3, 5], [3, 0], [3, 6], [5, 1], [4, 1], [2, 0], [5, 0], [5, 5]]
        fg = 13 # This is from final_pred.py, unclear what it represents but it's used there.
        if pl in l_cond_w1:
            if not (pts_cvzone[0][0] + fg < pts_cvzone[8][0] and pts_cvzone[0][0] + fg < pts_cvzone[12][0] and pts_cvzone[0][0] + fg < pts_cvzone[16][0] and
                    pts_cvzone[0][0] + fg < pts_cvzone[20][0]) and not (
                    pts_cvzone[0][0] > pts_cvzone[8][0] and pts_cvzone[0][0] > pts_cvzone[12][0] and pts_cvzone[0][0] > pts_cvzone[16][0] and pts_cvzone[0][0] > pts_cvzone[20][
                0]) and self.euclidean_distance(pts_cvzone[4], pts_cvzone[11]) < 50:
                ch1_idx = 1

        l_cond_w2 = [[5, 0], [5, 5], [0, 1]]
        if pl in l_cond_w2:
            if pts_cvzone[6][1] > pts_cvzone[8][1] and pts_cvzone[10][1] > pts_cvzone[12][1] and pts_cvzone[14][1] > pts_cvzone[16][1]:
                ch1_idx = 1

        # -------------------------condn for 8 groups ends, now apply subgroup refinements

        final_char = CNN_CHAR_GROUP_MAP.get(ch1_idx, " ") # Get the initial group character

        if ch1_idx == 0: # Group for S, A, T, E, M, N
            if pts_cvzone[4][0] < pts_cvzone[6][0] and pts_cvzone[4][0] < pts_cvzone[10][0] and pts_cvzone[4][0] < pts_cvzone[14][0] and pts_cvzone[4][0] < pts_cvzone[18][0]:
                final_char = 'A'
            elif pts_cvzone[4][0] > pts_cvzone[6][0] and pts_cvzone[4][0] < pts_cvzone[10][0] and pts_cvzone[4][0] < pts_cvzone[14][0] and pts_cvzone[4][0] < pts_cvzone[18][
                0] and pts_cvzone[4][1] < pts_cvzone[14][1] and pts_cvzone[4][1] < pts_cvzone[18][1]:
                final_char = 'T'
            elif pts_cvzone[4][1] > pts_cvzone[8][1] and pts_cvzone[4][1] > pts_cvzone[12][1] and pts_cvzone[4][1] > pts_cvzone[16][1] and pts_cvzone[4][1] > pts_cvzone[20][1]:
                final_char = 'E'
            elif pts_cvzone[4][0] > pts_cvzone[6][0] and pts_cvzone[4][0] > pts_cvzone[10][0] and pts_cvzone[4][0] > pts_cvzone[14][0] and pts_cvzone[4][1] < pts_cvzone[18][1]:
                final_char = 'M'
            elif pts_cvzone[4][0] > pts_cvzone[6][0] and pts_cvzone[4][0] > pts_cvzone[10][0] and pts_cvzone[4][1] < pts_cvzone[18][1] and pts_cvzone[4][1] < pts_cvzone[14][1]:
                final_char = 'N'
            else: # Default for group 0 if no specific condition met
                final_char = 'S'

        elif ch1_idx == 2: # Group for C, O
            if self.euclidean_distance(pts_cvzone[12], pts_cvzone[4]) > 42:
                final_char = 'C'
            else:
                final_char = 'O'

        elif ch1_idx == 3: # Group for G, H
            if (self.euclidean_distance(pts_cvzone[8], pts_cvzone[12])) > 72:
                final_char = 'G'
            else:
                final_char = 'H'

        elif ch1_idx == 7: # Group for Y, J
            if self.euclidean_distance(pts_cvzone[8], pts_cvzone[4]) > 42:
                final_char = 'Y'
            else:
                final_char = 'J'

        elif ch1_idx == 4: # Group for L
            final_char = 'L'

        elif ch1_idx == 6: # Group for X
            final_char = 'X'

        elif ch1_idx == 5: # Group for P, Q, Z
            if pts_cvzone[4][0] > pts_cvzone[12][0] and pts_cvzone[4][0] > pts_cvzone[16][0] and pts_cvzone[4][0] > pts_cvzone[20][0]:
                if pts_cvzone[8][1] < pts_cvzone[5][1]:
                    final_char = 'Z'
                else:
                    final_char = 'Q'
            else:
                final_char = 'P'

        elif ch1_idx == 1: # Group for B, D, F, I, W, K, U, V, R
            if (pts_cvzone[6][1] > pts_cvzone[8][1] and pts_cvzone[10][1] > pts_cvzone[12][1] and pts_cvzone[14][1] > pts_cvzone[16][1] and pts_cvzone[18][1] > pts_cvzone[20][1]):
                final_char = 'B'
            elif (pts_cvzone[6][1] > pts_cvzone[8][1] and pts_cvzone[10][1] < pts_cvzone[12][1] and pts_cvzone[14][1] < pts_cvzone[16][1] and pts_cvzone[18][1] < pts_cvzone[20][1]):
                final_char = 'D'
            elif (pts_cvzone[6][1] < pts_cvzone[8][1] and pts_cvzone[10][1] > pts_cvzone[12][1] and pts_cvzone[14][1] > pts_cvzone[16][1] and pts_cvzone[18][1] > pts_cvzone[20][1]):
                final_char = 'F'
            elif (pts_cvzone[6][1] < pts_cvzone[8][1] and pts_cvzone[10][1] < pts_cvzone[12][1] and pts_cvzone[14][1] < pts_cvzone[16][1] and pts_cvzone[18][1] > pts_cvzone[20][1]):
                final_char = 'I'
            elif (pts_cvzone[6][1] > pts_cvzone[8][1] and pts_cvzone[10][1] > pts_cvzone[12][1] and pts_cvzone[14][1] > pts_cvzone[16][1] and pts_cvzone[18][1] < pts_cvzone[20][1]):
                final_char = 'W'
            elif (pts_cvzone[6][1] > pts_cvzone[8][1] and pts_cvzone[10][1] > pts_cvzone[12][1] and pts_cvzone[14][1] < pts_cvzone[16][1] and pts_cvzone[18][1] < pts_cvzone[20][1]) and pts_cvzone[4][1] < pts_cvzone[9][1]:
                final_char = 'K'
            elif ((self.euclidean_distance(pts_cvzone[8], pts_cvzone[12]) - self.euclidean_distance(pts_cvzone[6], pts_cvzone[10])) < 8) and (
                    pts_cvzone[6][1] > pts_cvzone[8][1] and pts_cvzone[10][1] > pts_cvzone[12][1] and pts_cvzone[14][1] < pts_cvzone[16][1] and pts_cvzone[18][1] <
                    pts_cvzone[20][1]):
                final_char = 'U'
            elif ((self.euclidean_distance(pts_cvzone[8], pts_cvzone[12]) - self.euclidean_distance(pts_cvzone[6], pts_cvzone[10])) >= 8) and (
                    pts_cvzone[6][1] > pts_cvzone[8][1] and pts_cvzone[10][1] > pts_cvzone[12][1] and pts_cvzone[14][1] < pts_cvzone[16][1] and pts_cvzone[18][1] <
                    pts_cvzone[20][1]) and (pts_cvzone[4][1] > pts_cvzone[9][1]):
                final_char = 'V'
            elif (pts_cvzone[8][0] > pts_cvzone[12][0]) and (
                    pts_cvzone[6][1] > pts_cvzone[8][1] and pts_cvzone[10][1] > pts_cvzone[12][1] and pts_cvzone[14][1] < pts_cvzone[16][1] and pts_cvzone[18][1] <
                    pts_cvzone[20][1]):
                final_char = 'R'
            else:
                final_char = 'B' # Default for group 1

        # --- Specific blanking condition from final_pred.py ---
        # This condition, when met, forces the character to be " "
        if final_char in ['B', 'E', 'S', 'X', 'Y']: # Note: The '1' in original final_pred.py was likely referring to 'B'
            if (pts_cvzone[6][1] > pts_cvzone[8][1] and pts_cvzone[10][1] < pts_cvzone[12][1] and pts_cvzone[14][1] < pts_cvzone[16][1] and pts_cvzone[18][1] > pts_cvzone[20][1]):
                final_char = " " # Set to blank

        # "next" gesture from final_pred.py becomes "SPACE"
        if final_char in ['E', 'Y', 'B']: # These letters are often involved in space gesture
            if (pts_cvzone[4][0] < pts_cvzone[5][0]) and \
               (pts_cvzone[6][1] > pts_cvzone[8][1] and pts_cvzone[10][1] > pts_cvzone[12][1] and
                pts_cvzone[14][1] > pts_cvzone[16][1] and pts_cvzone[18][1] > pts_cvzone[20][1]):
                final_char = "SPACE"
        
        # Backspace gesture (logic from final_pred.py)
        # Assuming original intent for 'Next' was 'SPACE' and it applies if the determined `final_char` is one of these
        if final_char in ['SPACE', 'B', 'C', 'H', 'F', 'X']:
            if (pts_cvzone[0][0] > pts_cvzone[8][0] and pts_cvzone[0][0] > pts_cvzone[12][0] and
                pts_cvzone[0][0] > pts_cvzone[16][0] and pts_cvzone[0][0] > pts_cvzone[20][0]) and \
               (pts_cvzone[4][1] < pts_cvzone[8][1] and pts_cvzone[4][1] < pts_cvzone[12][1] and
                pts_cvzone[4][1] < pts_cvzone[16][1] and pts_cvzone[4][1] < pts_cvzone[20][1]) and \
               (pts_cvzone[4][1] < pts_cvzone[6][1] and pts_cvzone[4][1] < pts_cvzone[10][1] and
                pts_cvzone[4][1] < pts_cvzone[14][1] and pts_cvzone[4][1] < pts_cvzone[18][1]):
                final_char = 'BACKSPACE'

        return final_char


    # --- GUI Action Methods ---
    def speak_fun(self):
        text_to_speak = "".join(self.current_sentence) + self.cnn_current_word_buffer
        if text_to_speak.strip():
            self.speak_engine.say(text_to_speak.strip())
            self.speak_engine.runAndWait()
            self.status_label.config(text=f"Speaking: '{text_to_speak.strip()}'", fg="blue")
        else:
            self.status_label.config(text="Nothing to speak.", fg="red")

    def clear_fun(self):
        self.current_sentence = []
        self.cnn_current_word_buffer = ""
        self.lstm_keypoints_sequence = []
        self.last_lstm_word_prediction = ""
        self.current_symbol = " "
        self.current_lstm_word = " "
        self.word1 = self.word2 = self.word3 = self.word4 = " " # Clear suggestions

        # Reset CNN debouncing state
        self.prev_char = " "
        self.count = -1
        self.ten_prev_char = [" "] * 10

        self.status_label.config(text="Sentence Cleared!", fg="green")

    def action_suggest(self, suggestion_word):
        if suggestion_word.strip() != "":
            # When a suggestion is taken, finalize the current word buffer
            if self.cnn_current_word_buffer.strip():
                # If there's a current buffer, replace it with the suggestion
                self.current_sentence.append(suggestion_word.strip())
            else: 
                # If buffer is empty, attempt to replace the last word in the sentence
                # This needs careful handling to avoid removing too much or breaking words
                temp_sentence_str = "".join(self.current_sentence).strip()
                if temp_sentence_str:
                    last_space_idx = temp_sentence_str.rfind(" ")
                    if last_space_idx != -1:
                        # Replace the last word
                        reconstructed_sentence = temp_sentence_str[:last_space_idx+1] + suggestion_word.strip()
                    else:
                        # Only one word in sentence, replace it
                        reconstructed_sentence = suggestion_word.strip()
                    self.current_sentence = [reconstructed_sentence] # Reconstruct as a single element for simplicity
                else: # If sentence was completely empty
                    self.current_sentence.append(suggestion_word.strip())
            
            # After applying a suggestion, ensure a space follows and clear the buffer
            if self.current_sentence and self.current_sentence[-1] != " ":
                self.current_sentence.append(" ")
            self.cnn_current_word_buffer = ""
            self.word1 = self.word2 = self.word3 = self.word4 = " " # Clear suggestions
            self.status_label.config(text=f"Applied suggestion: '{suggestion_word}'", fg="green")

    def set_phrase_mode(self):
        if self.prediction_mode != 'phrase':
            self.prediction_mode = 'phrase'
            # Finalize any pending CNN buffer before switching
            if self.cnn_current_word_buffer.strip():
                self.current_sentence.append(self.cnn_current_word_buffer.strip())
                self.current_sentence.append(" ")
            self.cnn_current_word_buffer = ""
            # Reset CNN debouncing state when switching away from it
            self.prev_char = " "
            self.count = -1
            self.ten_prev_char = [" "] * 10

            self.last_lstm_word_prediction = "" # Reset LSTM for immediate prediction
            self.status_label.config(text="Mode: Phrase (LSTM)", fg="blue")
            print("Switched to Phrase (LSTM) Mode")

    def set_letter_mode(self):
        if self.prediction_mode != 'letter':
            self.prediction_mode = 'letter'
            self.lstm_keypoints_sequence = [] # Clear LSTM buffer
            self.last_lstm_word_prediction = "" # Reset LSTM word state

            # Reset CNN debouncing state when switching to it
            self.prev_char = " "
            self.count = -1
            self.ten_prev_char = [" "] * 10
            
            self.status_label.config(text="Mode: Letter (CNN)", fg="blue")
            print("Switched to Letter (CNN) Mode")

    def set_both_modes(self):
        if self.prediction_mode != 'both':
            self.prediction_mode = 'both'
            # Finalize any pending CNN buffer if switching from letter mode
            if self.cnn_current_word_buffer.strip():
                self.current_sentence.append(self.cnn_current_word_buffer.strip())
                self.current_sentence.append(" ")
            self.cnn_current_word_buffer = ""
            
            # Reset CNN debouncing state
            self.prev_char = " "
            self.count = -1
            self.ten_prev_char = [" "] * 10

            self.lstm_keypoints_sequence = [] # Clear LSTM buffer
            self.last_lstm_word_prediction = "" # Reset LSTM word state
            
            self.status_label.config(text="Mode: Both (LSTM prioritized)", fg="blue")
            print("Switched to Both Modes")

    def destructor(self):
        print("Closing Application...")
        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()
        self.holistic_detector.close()
        self.speak_engine.stop()

# --- Helper Functions (outside class, adapted from your provided files) ---

# From my_functions.py
def image_process_holistic(image, model):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.process(image_rgb)
    return results

# keypoint_extraction_holistic remains outside as it doesn't use self.mp_drawing
def keypoint_extraction_holistic(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    keypoints = np.concatenate([lh, rh])
    return keypoints

# --- Application Entry Point ---
if __name__ == "__main__":
    app = Application()
    app.root.mainloop()
