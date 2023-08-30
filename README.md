# Stormhacks22: Emotion Detector for ASL

Authors: [Sahaj Singh](https://github.com/SatireSage), [Rohan Karan](https://github.com/RohanKaran), [Nicolas Ramirez](https://github.com/Pikanick), [Boris Perdija](https://github.com/bperdija)

This project was created during the 2022 Stormhacks hackathon by Sahaj Singh, Rohan Karan, Nicolas Ramirez, and Boris Perdija.

## Inspiration

Approximately 12 million people aged 40 years and over in the United States have vision impairment, including 1 million who are blind. Consequently, these individuals are unable to perceive the facial and corporeal expressions that are instrumental in conveying emotions.

## What it does

Using advanced ML and computer vision techniques, our pre-trained program is adept at detecting ASL (American Sign Language) signs. It translates these signs in real-time, providing an immediate output to the user.

## How We Built It

We primarily employed TensorFlow for the training and creation of our model. Key detection of palms, hands, and faces was achieved using Mediapipe. The detected keypoints subsequently facilitated the prediction of human emotions.

## Challenges We Encountered

Our most significant challenge was the acquisition of an appropriate ASL dataset encompassing a diverse range of words, vital for the training of our ASL model.

## Accomplishments

We pride ourselves on crafting a model capable of predicting human emotion and body language with commendable accuracy.

## Learnings

The journey enlightened us on the intricacies of detecting human emotions, body language, and notably, American Sign Language, through computer vision.

## Future Directions for "Emotion Detector for ASL"

Our aspirations involve refining our ASL model by integrating a larger dataset, with the ultimate aim of enhancing its accuracy.

## Setting Up and Running the Program

To run this program:

1. Install the required dependencies:
   <pre><div class="bg-black rounded-md mb-4"><div class="flex items-center relative text-gray-200 bg-gray-800 px-4 py-2 text-xs font-sans justify-between rounded-t-md"><span>bash</span><button class="flex ml-auto gap-2"><svg stroke="currentColor" fill="none" stroke-width="2" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg"><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path><rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect></svg></button></div><div class="p-4 overflow-y-auto"><code class="!whitespace-pre hljs language-bash">pip install -r requirements.txt
   </code></div></div></pre>
2. Once the dependencies are in place, execute the `mp_keypoints_show.py` script:
   <pre><div class="bg-black rounded-md mb-4"><div class="flex items-center relative text-gray-200 bg-gray-800 px-4 py-2 text-xs font-sans justify-between rounded-t-md"><span>bash</span><button class="flex ml-auto gap-2"><svg stroke="currentColor" fill="none" stroke-width="2" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg"><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path><rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect></svg></button></div><div class="p-4 overflow-y-auto"><code class="!whitespace-pre hljs language-bash">python feed\mp_keypoints_show.py
   </code></div></div></pre>

## Built With

- Machine Learning
- Python

## Media

Check out our project demonstration on YouTube: [Emotion Detector for ASL - YouTube](https://www.youtube.com/watch?v=E5jAKVbiboY)

## More Information

For a detailed project overview, visit our submission on Devpost: [Emotion Detector - Devpost](https://devpost.com/software/emotion-detector-o60hev)

## License

This project is licensed under the MIT License.
