# audio_classification

## dataset

###     1. source dataset
    (1) drone driving audio(challenge)
        - drone noise
        - rescue request(human)
          - male / female / child ( 3 class )

    (2) drone noise(youtube)
        - drone sounds similar to challenge drones.
        - siren sound

###     2. processed dataset
    (1-1) only drone noise
          - drone driving audio without rescue request (using it to remove drone noise)

    (1-2) only drone noise + siren
          - drone driving audio without rescue request (using it to remove drone noise)

    (2) only rescue request
        - rescue request audio with drone noise (using it to classification)
    
    
    
## train

    (1) audio preprocessing
    (2) extract audio features
    (3) stft / mfcc 
    (4) neural network training 



## test

    (1) audio classification ( output : json file )
    - male : (# m # s)
    - female : (# m # s)
    - child : (# m # s)

