# Status options
uint8 READY = 0  # Prepared but not started
uint8 DRIVING = 1
uint8 COMPLETED = 2
uint8 FAILED = 3


# -------Msg------- #
uint8 state

float32 duration
float32 distance

float32 multiplicator
uint16 mistakes

float32 score

uint8 parking_successes
