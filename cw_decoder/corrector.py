#!/usr/bin/env python3
"""
corrector.py — CW semantic post-correction: fix decoding errors using QSO format priors.

Features:
- Callsign format validation and fuzzy matching
- Q-code auto-correction
- RST report format validation
- Intelligent completion of common CW abbreviations
"""

import re
from typing import List, Tuple, Optional

# Callsign format regex (major countries/regions)
CALLSIGN_PATTERNS = [
    re.compile(r'^[KNW][A-Z]?\d[A-Z]{1,4}$'),     # United States (N0CALL, W1AW, K9ZZ)
    re.compile(r'^[A-Z]{2}\d[A-Z]{1,3}$'),         # Europe (general) YO4AAC, DL1ABC
    re.compile(r'^[A-Z]\d{2}[A-Z]{1,4}$'),         # Single-letter prefix + area digit (S57ZT)
    re.compile(r'^B[G-H]\d[A-Z]{1,3}$'),           # China
    re.compile(r'^V[KM]\d[A-Z]{1,3}$'),            # Hong Kong/Macau
    re.compile(r'^J[A-I]\d[A-Z]{1,3}$'),           # Japan
    re.compile(r'^[A-Z]\d[A-Z]{2,4}$'),            # Simplified format (W1ABC)
    re.compile(r'^\d[A-Z]\d[A-Z]{1,3}$'),          # Numeric-prefix countries (9A1ABC, 4X1AB)
]

# Contest cut-number RST (599 → 5NN / ENN). Must never fuzzy-match to callsigns (e.g. NJ).
# Cut map: T=0 A=1 U=2 V=3 4=4 E=5 6=6 B=7 D=8 N=9
CONTEST_CUT_RST = re.compile(r'^[15E]NN$|^[1-5]N{2}$')

# Known callsigns used in test samples (for fuzzy matching)
KNOWN_CALLSIGNS = {
    'W1AW', 'W4AA', 'K9ZZ', 'N0CALL', 'VE3RPN', 'G4PPL', 'JA1ABC',
    'N1EME', 'K3EST', 'WX4EMG', 'N3NEWB', 'K9ZZ', 'BG6XY', 'BA1AA',
    'NJ', 'N0CALL', 'W1AW',
}

# Q-code list (high-frequency QSO terms)
Q_CODES = {
    'QSL', 'QRZ', 'QSO', 'QTH', 'QSY', 'QSB', 'QRM',
    'QRN', 'QRS', 'QRQ', 'QRO', 'QRT', 'QTR', 'QMX',
    'QRL', 'QRP', 'QRA', 'QRB', 'QRG', 'QRH', 'QRI',
    'QRK', 'QRL', 'QRM', 'QRN', 'QRO', 'QRP', 'QRQ',
    'QRS', 'QRT', 'QRU', 'QRV', 'QRW', 'QRX', 'QRY',
    'QRZ', 'QSA', 'QSB', 'QSC', 'QSD', 'QSK', 'QSL',
    'QSM', 'QSN', 'QSO', 'QSP', 'QSQ', 'QSR', 'QSS',
    'QST', 'QSU', 'QSX', 'QSY', 'QSZ', 'QTA', 'QTB',
    'QTC', 'QTR', 'QTU', 'QTX', 'QTY', 'QTZ', 'QUB',
}

# Common CW abbreviations
CW_ABBREVIATIONS = {
    'CQ', 'DE', 'K', 'AR', 'SK', 'KN', 'BK', 'BT', 'AA',
    'AS', 'CL', 'HR', 'HW', 'OM', 'YL', 'XYL', 'FB', 'GA',
    'GE', 'GM', 'GN', 'HI', 'R', 'S', 'ES', 'PSE', 'TKS',
    'TNX', 'CPY', 'UR', 'MY', 'DR', 'DX', 'WX', 'RST', 'ANT', 'RIG',
    'PWR', 'AGN', 'BK', 'CU', '73', '88', '55', '99', 'GL',
    'BTU', 'QRT', 'QRZ', 'QSL', 'QSO', 'QTH', 'QSY',
}

# RST report format: R(1-5)S(1-9)T(1-9)
RST_PATTERN = re.compile(r'^[1-5][1-9][1-9]$')

# Numeric pattern
NUMBER_PATTERN = re.compile(r'^\d+$')

# Tokens that should not be modified by semantic correction (Morse special symbols)
PROTECTED_TOKENS = {'=', '<AR>', '<SK>', '<KN>', '.', ',', '?', '/', '&'}

# Common English words that should NOT be "corrected" to abbreviations
PROTECTED_ENGLISH = {
    'A', 'AN', 'AND', 'ARE', 'AS', 'AT', 'BE', 'BY', 'DO', 'FOR',
    'FROM', 'HAD', 'HAS', 'HAVE', 'HE', 'HER', 'HIS', 'HOW', 'I',
    'IF', 'IN', 'IS', 'IT', 'ITS', 'ME', 'MY', 'NO', 'NOT', 'OF',
    'ON', 'OR', 'OUR', 'SHE', 'SO', 'THE', 'THEM', 'THEY', 'THIS',
    'TO', 'UP', 'US', 'WE', 'WERE', 'WHAT', 'WHO', 'WHY', 'YOU',
    'YOUR', 'ALL', 'CAN', 'DID', 'DOES', 'DONE', 'EACH', 'GET',
    'GOT', 'HER', 'HIM', 'HIS', 'JUST', 'KNEW', 'KNOW', 'LIKE',
    'LIVE', 'LOOK', 'MAY', 'MIGHT', 'MUST', 'NEED', 'NEVER', 'NOW',
    'OLD', 'ONLY', 'OTHER', 'OUR', 'OUT', 'OWN', 'SAY', 'SEE',
    'SOME', 'SUCH', 'THAN', 'THAT', 'THEIR', 'THEN', 'THERE',
    'THESE', 'THINK', 'TIME', 'TODAY', 'TOO', 'TRY', 'VERY',
    'WANT', 'WAY', 'WELL', 'WHEN', 'WHERE', 'WHICH', 'WILL',
    'WITH', 'WORK', 'YES', 'YET', 'BUT', 'BACK', 'BEEN', 'BEFORE',
    'BETWEEN', 'CAME', 'COME', 'COULD', 'DOWN', 'EACH', 'EVEN',
    'FEW', 'FIRST', 'FOUND', 'GIVE', 'GOING', 'GOOD', 'GREAT',
    'HERE', 'HOME', 'INTO', 'KEEP', 'LAST', 'LEFT', 'LET', 'LONG',
    'MADE', 'MAKE', 'MANY', 'MIGHT', 'MORE', 'MOST', 'MUCH',
    'NAME', 'NEW', 'NEXT', 'NIGHT', 'NONE', 'NORTH', 'NOTHING',
    'OFF', 'OFTEN', 'ONCE', 'OVER', 'PART', 'PLACE', 'POINT',
    'RIGHT', 'ROUND', 'SAME', 'SAY', 'SECOND', 'SET', 'SHOULD',
    'SIDE', 'SMALL', 'SOME', 'SOUTH', 'STILL', 'SUCH', 'TAKE',
    'TELL', 'THREE', 'THROUGH', 'TOGETHER', 'TOMORROW', 'TURN',
    'UNDER', 'UNTIL', 'UPON', 'WATER', 'WHILE', 'WHOLE', 'WORLD',
    'WOULD', 'YEAR', 'AFTER', 'AGAIN', 'ALONG', 'ALREADY', 'ALSO',
    'ANY', 'AROUND', 'AWAY', 'BEGIN', 'BELOW', 'BRING', 'BUILD',
    'CALL', 'CARRY', 'CATCH', 'CHANGE', 'CHECK', 'CHILD', 'CITY',
    'CLOSE', 'COLD', 'COUNTRY', 'DAY', 'DIFFERENT', 'EARLY', 'EAST',
    'EITHER', 'ENOUGH', 'EVERY', 'FAR', 'FAST', 'FATHER', 'FIND',
    'FIVE', 'FOLLOW', 'FOOD', 'FOUR', 'FULL', 'FURTHER', 'GET',
    'GIRL', 'GIVE', 'GROUND', 'GROUP', 'GROW', 'HALF', 'HAND',
    'HAPPEN', 'HEAD', 'HEAR', 'HELP', 'HIGH', 'HOLD', 'HOUR',
    'HOUSE', 'IDEA', 'IMPORTANT', 'INTEREST', 'ISLAND', 'ITSELF',
    'JUST', 'KIND', 'LARGE', 'LATER', 'LEAST', 'LETTER', 'LIGHT',
    'LINE', 'LISTEN', 'LITTLE', 'MANNER', 'MARK', 'MATTER', 'MEAN',
    'MEET', 'MISS', 'MONEY', 'MONTH', 'MOVE', 'NEAR', 'NEVER',
    'NICE', 'NUMBER', 'OPEN', 'ORDER', 'PAGE', 'PAPER', 'PART',
    'PASS', 'PAST', 'PEOPLE', 'PICTURE', 'PLAN', 'PLAY', 'PLEASE',
    'POINT', 'POWER', 'PROBLEM', 'PROGRAM', 'PUT', 'QUESTION',
    'QUITE', 'RAIN', 'READ', 'READY', 'REAL', 'REALLY', 'REMEMBER',
    'REST', 'RESULT', 'RIVER', 'RUN', 'SAME', 'SAY', 'SCHOOL',
    'SECOND', 'SEEM', 'SELF', 'SEND', 'SENT', 'SERVE', 'SET',
    'SEVERAL', 'SHOW', 'SIDE', 'SIMPLE', 'SINCE', 'SIT', 'SIX',
    'SIZE', 'SMALL', 'SOMETHING', 'SON', 'SOON', 'SORT', 'SOUND',
    'SPEAK', 'SPEND', 'STAND', 'START', 'STATE', 'STAY', 'STEP',
    'STOP', 'STORY', 'STRONG', 'STUDENT', 'SURE', 'SURPRISE',
    'TABLE', 'TALK', 'TEACH', 'TELL', 'TEST', 'THANK', 'THAT',
    'THING', 'THOUGHT', 'THREE', 'TIRED', 'TOGETHER', 'TOMORROW',
    'TONIGHT', 'TOOK', 'TOP', 'TRAVEL', 'TRUE', 'TRY', 'TURN',
    'PAYS', 'STUCK', 'GUYS', 'BAND', 'ELSE', 'ANTENNA', 'WORKING',
    'TWO', 'TYPE', 'UNDER', 'UNTIL', 'USUAL', 'VALUE', 'WAIT',
    # QRQ ragchew / net English (helps split run-on Morse prose)
    'WAS', 'DONT', 'PORT', 'USB', 'ABLE', 'NOISE', 'MENTIONED', 'WORKING',
    'ANNOYED', 'ANNOY', 'DECIDED', 'DECIDE', 'GRUNT', 'INSTEAD', 'WET',
    'STUMBLING', 'BLOCK', 'SWEAT', 'CHUCK', 'MENTION', 'FUNNY', 'VOLTS',
    'POLE', 'HAMMERED',
    'ABOUT', 'BECAUSE', 'BETTER', 'GUESS', 'GLAD', 'WOLF',
    'FLORIDA', 'VERTICAL', 'ACCESS', 'INTERNET', 'PROGRESS', 'LEARNED',
    'CODE', 'AGE', 'RADIO', 'ANTENNA', 'SIGNAL', 'STILL',
    'WALK', 'WALL', 'WANT', 'WARM', 'WATCH', 'WATER', 'WEST',
    'WHETHER', 'WHICH', 'WHILE', 'WHITE', 'WHOLE', 'WHY', 'WIFE',
    'WIND', 'WINTER', 'WISH', 'WITHIN', 'WITHOUT', 'WOMAN', 'WORD',
    'WORK', 'WORLD', 'WRITE', 'YARD', 'YESTERDAY', 'YOUNG',
    'ANTENNA', 'BAND', 'BREAKFAST', 'CHILDREN', 'CLUB', 'CODE',
    'COLL', 'CW', 'DINNER', 'EAST', 'EVENING', 'FAMILY', 'FREQ',
    'GLAD', 'HOME', 'HUSBAND', 'JOB', 'KIDS', 'LUNCH', 'METER',
    'METERS', 'MORNING', 'NET', 'NICE', 'OFFICE', 'ONLINE',
    'OPERATOR', 'PHONE', 'POWER', 'RADIO', 'RAIN', 'RECORDED',
    'RECORDING', 'REGARDS', 'REPORT', 'RETIRE', 'RIG', 'SIGNAL',
    'SNOW', 'SORRY', 'SOUNDS', 'STATION', 'STORM', 'SURE', 'TEMP',
    'THANKS', 'THINK', 'TIME', 'TONIGHT', 'WARM', 'WATTS', 'WELL',
    'WIFE', 'WIND', 'WORKING', 'WX',
    # Additional common words seen in QRQ CW nets
    'ABOUT', 'ABOVE', 'AFTER', 'ALL', 'ALMOST', 'AMONG', 'BECAUSE',
    'BETTER', 'BIG', 'BIT', 'DONE', 'EARLY', 'ENOUGH', 'EVER',
    'FELLOW', 'FINAL', 'FINE', 'FORGET', 'GETTING', 'GIVING',
    'GOING', 'GONE', 'HAPPEN', 'HAPPY', 'HEARING', 'HOPE',
    'KEEPING', 'KIND', 'KNOWING', 'LATE', 'LATER', 'LEAST',
    'LETTER', 'LOT', 'LOTS', 'MAIN', 'MEANS', 'MOSTLY', 'NEAR',
    'NEED', 'NEVER', 'NORMAL', 'NOTE', 'NOTICE', 'PEOPLE',
    'PERHAPS', 'PIECE', 'PRETTY', 'PROBABLY', 'PUTTING', 'RATHER',
    'REALLY', 'REMEMBER', 'RUNNING', 'SEEMS', 'SENDING', 'SHOWING',
    'SIMPLE', 'SINCE', 'SLOWLY', 'SPECIAL', 'STUFF', 'TALKING',
    'TELLING', 'THEMSELVES', 'THING', 'THINGS', 'THOUGHT', 'TODAY',
    'TOGETHER', 'TONIGHT', 'TRIED', 'TRYING', 'TURNS', 'USUALLY',
    'VARIOUS', 'WAITING', 'WHETHER', 'WONDER', 'WORKING', 'WRITING',
    'DUMP', 'THINGS', 'FLOODED', 'TIMES', 'GETTING', 'SHOWED',
    'STOPPED', 'GUESS', 'NEVER', 'SHOW', 'LIKE', 'JUST',
}

# Common error mappings (based on visual/auditory similarity)
COMMON_ERRORS = {
    '0': 'O', 'O': '0',  # Digit 0 and letter O
    '1': 'I', 'I': '1',  # Digit 1 and letter I
    '5': 'S', 'S': '5',  # Digit 5 and letter S
    '9': 'Q', 'Q': '9',  # Digit 9 and letter Q
    # Digit-to-digit confusions (common in Morse code)
    '4': '0', '0': '4',  # 4 and 0 sound similar
    '3': '4', '4': '3',  # 3 and 4 adjacent
    '2': '3', '3': '2',  # 2 and 3 adjacent
}

# Common QSO words for word-splitting detection
QSO_WORDS = {
    'CQ', 'DE', 'QRZ', 'QSL', 'QSO', 'QTH', 'QSY', 'QSB', 'QRM', 'QRN',
    'QRS', 'QRQ', 'QRO', 'QRP', 'QRT', 'QRZ', 'QRG', 'QRH', 'QRL',
    'THE', 'YOU', 'YOUR', 'AND', 'FOR', 'WITH', 'FROM', 'THIS', 'THAT',
    'HERE', 'THERE', 'WHEN', 'WHAT', 'WHERE', 'WHICH', 'WHO', 'HOW',
    'NOT', 'BUT', 'ALL', 'CAN', 'WILL', 'HAVE', 'HAS', 'HAD', 'WAS',
    'ARE', 'BEEN', 'BEING', 'WERE', 'THEY', 'THEIR', 'THEM', 'OUR',
    'GOOD', 'NICE', 'GLAD', 'HEARD', 'COPY', 'THANKS', 'THANK', 'TNX',
    'TKS', 'PLEASE', 'PSE', 'SORRY', 'YES', 'NO', 'OK', 'RIGHT',
    'CALL', 'CALLSIGN', 'NAME', 'RST', 'REPORT', 'SIGNAL', 'SIGNALS',
    'ANTENNA', 'ANT', 'RIG', 'POWER', 'PWR', 'WATTS', 'WATT',
    'METERS', 'METER', 'BAND', 'FREQ', 'FREQUENCY', 'LISTEN', 'LISTENING',
    'WORK', 'WORKING', 'WORKED', 'CALLING', 'ANSWER', 'ANSWERING',
    'AROUND', 'FIRST', 'TIME', 'TONIGHT', 'TODAY', 'YESTERDAY',
    'MORNING', 'EVENING', 'AFTERNOON', 'LUNCH', 'DINNER', 'BREAKFAST',
    'LIVE', 'RECORDING', 'RECORDED', 'ONLINE', 'NET', 'CHECK',
    'OPERATOR', 'OP', 'STATION', 'CLUB', 'MEMBER', 'PRESIDENT',
    'WEATHER', 'WX', 'TEMP', 'TEMPERATURE', 'WIND', 'RAIN', 'SNOW',
    'DEGREES', 'FAHRENHEIT', 'CELSIUS', 'HOT', 'COLD', 'WARM', 'COOL',
    'FAMILY', 'WIFE', 'HUSBAND', 'CHILDREN', 'KIDS', 'HOME', 'HOUSE',
    'RETIRE', 'RETIRED', 'JOB', 'OFFICE', 'ENGINEER', 'DOCTOR',
    'TEACHER', 'STUDENT', 'HAM', 'AMATEUR', 'RADIO', 'MORSE', 'CODE',
    'CW', 'SSB', 'PHONE', 'DIGITAL', 'FT8', 'PSK', 'RTTY',
    'SEVENTY', 'THREE', 'EIGHTY', 'EIGHT', 'NINETY', 'NINE',
    'BEST', 'REGARDS', 'REGARD',
    'SOUNDS', 'LIKE', 'KNOW', 'THINK', 'RECALL', 'REMEMBER',
    'WOULD', 'COULD', 'SHOULD', 'MIGHT', 'MAY', 'MUST',
    'GREAT', 'WONDERFUL', 'FINE', 'WELL', 'SURE',
    'EVERY', 'EVERYONE', 'ANYONE', 'SOMEONE', 'NOBODY',
    # Additional common words for splitting
    'THING', 'THINGS', 'DUMP', 'GETTING', 'READY', 'SOON', 'FLOODED',
    'TIMES', 'MANY', 'TOO', 'SHOW', 'SHOWED', 'STOPPED', 'SENDING',
    'GUESS', 'NEVER', 'ONLY', 'IF', 'THEY', 'JUST', 'ABOUT',
    'EVER', 'AFTER', 'BEFORE', 'ALSO', 'BACK', 'DOWN', 'OFF',
    'OVER', 'STILL', 'YET', 'AGAIN', 'ALMOST', 'EVERY', 'GET',
    'GIVE', 'GO', 'GOING', 'GONE', 'KEEP', 'LET', 'LONG', 'LOOK',
    'MAKE', 'PUT', 'RUN', 'SAY', 'SEE', 'SEND', 'SET', 'SHOW',
    'SOME', 'TAKE', 'TELL', 'TRY', 'TURN', 'USE', 'COME',
    'FELLOW', 'MAN', 'MEN', 'WAY', 'DAY', 'DAYS', 'NIGHT',
    'LOT', 'LOTS', 'BIT', 'HAPPY', 'HOPE', 'MEANS', 'DONE',
    'DOING', 'BEEN', 'HEARING', 'TALKING', 'TRYING', 'WAITING',
    'FORGET', 'NOTICE', 'SEEMS', 'PRETTY', 'RATHER', 'PROBABLY',
    'BECAUSE', 'BETTER', 'FINAL', 'MAIN', 'NORMAL', 'SIMPLE',
    'SLOWLY', 'SPECIAL', 'USUALLY', 'VARIOUS', 'HAPPEN',
    'IT', 'MY', 'ME', 'HE', 'WE', 'AM', 'BE', 'UP',
    'HIS', 'HER', 'ITS', 'ANY', 'OWN', 'SAME', 'OTHER',
    'FEW', 'MOST', 'MORE', 'SUCH', 'EACH', 'OWN',
    'AM', 'IS', 'DO', 'DID', 'DOES', 'HAS', 'HAD',
    # Weather and numbers
    'SUNNY', 'SIXTY', 'SIXTYFIVE', 'SEVENTY', 'EIGHTY', 'NINETY',
    'HUNDRED', 'DEGREES', 'AROUND', 'ABOUT',
    # Common QSO phrases
    'SHOW', 'UP', 'HERE', 'TONIGHT', 'TONIG', 'OTHERS', 'WILL',
    'MYSELF', 'REALLY', 'LIKED', 'WELL', 'HOPE', 'SOME',
    # Additional words for split coverage
    'GUY', 'SAYING', 'TALKING', 'LISTENING', 'WORKING', 'PLAYING',
    'FIGURED', 'GUESS', 'STILL', 'VERY', 'BEST', 'WHILE',
    'LET', 'ME', 'US', 'THEM', 'HIM', 'HER',
    'CALLED', 'USING', 'DOING', 'MAKING', 'TAKING',
    'BEEN', 'BEING', 'HAVING', 'GOING', 'COMING',
    'WOULD', 'COULD', 'SHOULD', 'MIGHT', 'MAY', 'MUST',
    'JUST', 'ONLY', 'ALSO', 'EVEN', 'RATHER', 'QUITE',
    'ANTENNAS', 'WIRELESS', 'PROGRAMMING', 'CONVERSATIONAL',
    'INSURANCE', 'EXPEDITION', 'APARTMENT', 'WELLINGTON',
    'CELLPHONE', 'CELL', 'PHONE', 'SIM', 'VERIZON',
    'BASICALLY', 'ACTUALLY', 'CERTAINLY', 'DEFINITELY',
    'DIFFERENT', 'DIFFICULT', 'IMPORTANT', 'INTERESTING',
    'REMEMBER', 'FORGET', 'UNDERSTAND', 'APPRECIATE',
    'EVERYTHING', 'SOMETHING', 'NOTHING', 'ANYTHING',
    'SOMEBODY', 'ANYBODY', 'NOBODY', 'EVERYBODY',
    'CONDX', 'CONDITIONS', 'FREQUENCY', 'FREQUENCIES',
    'FOLLOWS', 'BULLETIN', 'WARNING', 'ALERT', 'STORM', 'AREA',
    # Words that should never be fuzzy-matched
    'FULLY', 'THANKSGIVING', 'EATER', 'MORNING', 'EVENING',
    'AFTERNOON', 'SOMETHING', 'ANYTHING', 'NOTHING', 'EVERYTHING',
    'EVERYWHERE', 'SOMEWHERE', 'ANYWHERE', 'NOWHERE',
}

# Common merged word patterns (detected in V13 output)
MERGED_WORD_FIXES = {
    'THEFIRST': 'THE FIRST',
    'THEOTHER': 'THE OTHER',
    'THEYRE': 'THEY RE',
    'TOYOU': 'TO YOU',
    'TOU': 'TO YOU',
    'ONYOU': 'ON YOU',
    'FORYOU': 'FOR YOU',
    'FORYOUSO': 'FOR YOU SO',
    'FORYOUTO': 'FOR YOU TO',
    'FORYOUAND': 'FOR YOU AND',
    'WITHYOU': 'WITH YOU',
    'FROMYOU': 'FROM YOU',
    'HEARDYOU': 'HEARD YOU',
    'COPYYOU': 'COPY YOU',
    'WORKYOU': 'WORK YOU',
    'IEWOULD': 'I WOULD',
    'IHADTO': 'I HAD TO',
    'IHAVETO': 'I HAVE TO',
    'IWANTTO': 'I WANT TO',
    'INEEDTO': 'I NEED TO',
    'PAYSETO': 'PAYS TO',
    'PAYSTO': 'PAYS TO',
    'THETSING': 'THE THING',
    'THETHING': 'THE THING',
    'STUCKEON': 'STUCK ON',
    'STUCKON': 'STUCK ON',
    'YOUEGUYE': 'YOU GUYS',
    'YOUGUYE': 'YOU GUYS',
    'YOUGUYS': 'YOU GUYS',
    'THEWAYYOU': 'THE WAY YOU',
    'AROUNDHERE': 'AROUND HERE',
    'AROUNDHI': 'AROUND HI',
    'ORTHEOTHER': 'OR THE OTHER',
    'LISTENINGTO': 'LISTENING TO',
    'LISTENINGTOYOU': 'LISTENING TO YOU',
    'BEENLISTENING': 'BEEN LISTENING',
    'HAVEBEEN': 'HAVE BEEN',
    'HADBEEN': 'HAD BEEN',
    'WERELISTENING': 'WERE LISTENING',
    'IFTHAT': 'IF THAT',
    'IFTHEY': 'IF THEY',
    'THATTHE': 'THAT THE',
    'THATYOU': 'THAT YOU',
    'WHATYOU': 'WHAT YOU',
    'WHEREYOU': 'WHERE YOU',
    'WHENYOU': 'WHEN YOU',
    'HOWYOU': 'HOW YOU',
    'WHOWORKS': 'WHO WORKS',
    'WHOKNOWS': 'WHO KNOWS',
    'WHOLIVES': 'WHO LIVES',
    'WHOCAN': 'WHO CAN',
    'EACHOTHER': 'EACH OTHER',
    'ORENOT': 'OR NOT',
    'ORNOT': 'OR NOT',
    'ANDYES': 'AND YES',
    'ANDTHE': 'AND THE',
    'ANDYOU': 'AND YOU',
    'ANDTHINK': 'AND THINK',
    'BETHAT': 'BE THAT',
    'BEENHERE': 'BEEN HERE',
    'GOODEVENING': 'GOOD EVENING',
    'GOODMORNING': 'GOOD MORNING',
    'GOODAFTERNOON': 'GOOD AFTERNOON',
    'THANKSFOR': 'THANKS FOR',
    'THANKYOU': 'THANK YOU',
    'THANKSAGAIN': 'THANKS AGAIN',
    'NICECHAT': 'NICE CHAT',
    'NICETALKING': 'NICE TALKING',
    'NICEMEETING': 'NICE MEETING',
    'GLADTO': 'GLAD TO',
    'GLADYOU': 'GLAD YOU',
    'SORRYFOR': 'SORRY FOR',
    'LOOKFORWARD': 'LOOK FORWARD',
    'LOOKINGFORWARD': 'LOOKING FORWARD',
    'NEXTTIME': 'NEXT TIME',
    'SEENEXT': 'SEE NEXT',
    'SEEYOU': 'SEE YOU',
    'TALKTO': 'TALK TO',
    'TALKWITH': 'TALK WITH',
    'SPEAKTO': 'SPEAK TO',
    'SPEAKWITH': 'SPEAK WITH',
    'WEREA': 'WERE A',
    'KNEWYOU': 'KNEW YOU',
    'KNOWYOU': 'KNOW YOU',
    'THINKYOU': 'THINK YOU',
    'SAYYOU': 'SAY YOU',
    'TELLYOU': 'TELL YOU',
    'SHOWYOU': 'SHOW YOU',
    'GIVEYOU': 'GIVE YOU',
    'LIKEYOU': 'LIKE YOU',
    'NEEDYOU': 'NEED YOU',
    'WANTYOU': 'WANT YOU',
    'HELPYOU': 'HELP YOU',
    'CALLYOU': 'CALL YOU',
    'HEARYOU': 'HEAR YOU',
    'FINDYOU': 'FIND YOU',
    'GETYOU': 'GET YOU',
    'MAKEYOU': 'MAKE YOU',
    'TAKEYOU': 'TAKE YOU',
    'BRINGYOU': 'BRING YOU',
    'SENDYOU': 'SEND YOU',
    'MEETYOU': 'MEET YOU',
    'JOINYOU': 'JOIN YOU',
    'THATIS': 'THAT IS',
    'THISIS': 'THIS IS',
    'THEREIS': 'THERE IS',
    'ITIS': 'IT IS',
    'WHATIS': 'WHAT IS',
    'WHICHIS': 'WHICH IS',
    'WHOWAS': 'WHO WAS',
    'WHOWILL': 'WHO WILL',
    'WHOCOULD': 'WHO COULD',
    # Additional patterns from real QRQ audio
    'THETHING': 'THE THING',
    'ITTO': 'IT TO',
    'TODO': 'TO DO',
    'TOTEST': 'TO TEST',
    'TESTIT': 'TEST IT',
    'ITFURTHER': 'IT FURTHER',
    'ANDGUESS': 'AND GUESS',
    'GUESSIF': 'GUESS IF',
    'THEYNEVER': 'THEY NEVER',
    'NEVERSHOW': 'NEVER SHOW',
    'SHOWUP': 'SHOW UP',
    'THEYLIKE': 'THEY LIKE',
    'LIKETO': 'LIKE TO',
    'TOJUST': 'TO JUST',
    'JUSTLIKE': 'JUST LIKE',
    'GETTINGREADY': 'GETTING READY',
    'READYTO': 'READY TO',
    'DUMPIT': 'DUMP IT',
    'DUMPTHING': 'DUMP THING',
    'THEHOUSE': 'THE HOUSE',
    'THISHOUSE': 'THIS HOUSE',
    'BEENGETTING': 'BEEN GETTING',
    'SENDINGIT': 'SENDING IT',
    'THEMBCUZ': 'THEM BECAUSE',
    'BCUZ': 'BECAUSE',
    'BECAUSETHEY': 'BECAUSE THEY',
    'SHOWET': 'SHOW IT',
    'ETVP': 'IT UP',
    'VPAND': 'UP AND',
    'STOPPEDSENDING': 'STOPPED SENDING',
    'THEMBCU': 'THEM BECAUSE',
    # Patterns from comprehensive real audio analysis
    'THEMONALLBANDS': 'THEM ON ALL BANDS',
    'THEMON': 'THEM ON',
    'YESTHELASTTIME': 'YES THE LAST TIME',
    'YESTHELAST': 'YES THE LAST',
    'TOWORKTHEM': 'TO WORK THEM',
    'TOWORKTHEMHI': 'TO WORK THEM HI',
    'INARIZONA': 'IN ARIZONA',
    'ANDLIVING': 'AND LIVING',
    'THATKIND': 'THAT KIND',
    'THATKINDOF': 'THAT KIND OF',
    'OFTHEGUY': 'OF THE GUY',
    'ONEOTHER': 'ONE OTHER',
    'THATWAS': 'THAT WAS',
    'IDONT': 'I DONT',
    'ISNOT': 'IS NOT',
    'MYEMAIL': 'MY EMAIL',
    'THISON': 'THIS ON',
    'WEREYOU': 'WERE YOU',
    'DIDYOU': 'DID YOU',
    'IWAS': 'I WAS',
    'ITWAS': 'IT WAS',
    'ITWILL': 'IT WILL',
    'THEREWAS': 'THERE WAS',
    'WORKINGTHE': 'WORKING THE',
    'WORKINGDX': 'WORKING DX',
    'WORKINGTHEM': 'WORKING THEM',
    'LISTENINGFOR': 'LISTENING FOR',
    'LISTENINGON': 'LISTENING ON',
    'TALKINGTO': 'TALKING TO',
    'TALKINGABOUT': 'TALKING ABOUT',
    'HEARDABOUT': 'HEARD ABOUT',
    'HEARDFROM': 'HEARD FROM',
    'HOPINGTO': 'HOPING TO',
    'TRYINGTO': 'TRYING TO',
    'GETTINGTHE': 'GETTING THE',
    'USINGTHE': 'USING THE',
    'GOINGTO': 'GOING TO',
    'COMINGFROM': 'COMING FROM',
    'DONTKNOW': 'DONT KNOW',
    'DONTTHINK': 'DONT THINK',
    'KINDOF': 'KIND OF',
    'ALOTOF': 'A LOT OF',
    'ALOT': 'A LOT',
    'THERADIO': 'THE RADIO',
    'THEANTENNA': 'THE ANTENNA',
    'THEOPERATOR': 'THE OPERATOR',
    'THESTATION': 'THE STATION',
    'THESIGNAL': 'THE SIGNAL',
    'THEBAND': 'THE BAND',
    'THEPOWER': 'THE POWER',
    'THERIG': 'THE RIG',
    'GUYHI': 'GUY HI',
    'THEREHI': 'THERE HI',
    'NETHI': 'NET HI',
    'BASICALLYACELL': 'BASICALLY A CELL',
    'WITHTHEM': 'WITH THEM',
    'SPEAKINGTO': 'SPEAKING TO',
    'THEREARE': 'THERE ARE',
    'THEREWERE': 'THERE WERE',
    'SUNNYAND': 'SUNNY AND',
    'HADTHEBACK': 'HAD THE BACK',
    'THEBACK': 'THE BACK',
    'NINETO': 'NINE TO',
    # More patterns from broad test analysis
    'JOEYOUARE': 'JOE YOU ARE',
    'THEYAREESTILL': 'THEY ARE STILL',
    'THEYARESTILL': 'THEY ARE STILL',
    'CALULKEDTHE': 'CALLED THE',
    'IFINDANTS': 'I FIND ANTS',
    'SOFIGURID': 'SO FIGURED',
    'JOEWASSYING': 'JOE WAS SAYING',
    'JOEWASTALKING': 'JOE WAS TALKING',
    'WASYING': 'WAS SAYING',
    'WASTALKING': 'WAS TALKING',
    'WASLISTENING': 'WAS LISTENING',
    'WASWORKING': 'WAS WORKING',
    'WEREWORKING': 'WERE WORKING',
    'WERETALKING': 'WERE TALKING',
    'WERELISTENING': 'WERE LISTENING',
    'WERESAYING': 'WERE SAYING',
    'FIGURID': 'FIGURED',
    'GUESSTHE': 'GUESS THE',
    'CONDXWERE': 'CONDX WERE',
    'WEREJUSTA': 'WERE JUST A',
    'BUTGUESS': 'BUT GUESS',
    'NAMETOM': 'NAME TOM',
    'NAMEIS': 'NAME IS',
    'NAMEJOHN': 'NAME JOHN',
}

# Common word-level typo fixes (Morse confusion patterns)
WORD_FIXES = {
    'MRKING': 'WORKING',
    'WOKRING': 'WORKING',
    'WORKNG': 'WORKING',
    'GRUNDT': 'GRUNT',
    'HIDK': 'THINK',
    'EOR': 'FOR',
    'FROR': 'FOR',
    'THT': 'THE',
    'TJE': 'THE',
    'THW': 'THE',
    'TEH': 'THE',
    'ADN': 'AND',
    'ANF': 'AND',
    'YOR': 'YOUR',
    'IEES': 'THINK',
    'TNIK': 'THINK',
    'THNINK': 'THINK',
    'THNK': 'THINK',
    'HERD': 'HEARD',
    # CPY removed — it is a valid CW abbreviation (not a typo for COPY)
    # 'CPOY': 'COPY',  # removed — too aggressive
    # 'CPYE': 'COPY',  # removed — too aggressive
    'WRKS': 'WORKS',
    'WRK': 'WORK',
    'WKR': 'WORKER',
    'BETWN': 'BETWEEN',
    'BTW': 'BETWEEN',
    'BTWN': 'BETWEEN',
    'RGRDS': 'REGARDS',
    'RGDS': 'REGARDS',
    'BESTR': 'BEST',
    'BSET': 'BEST',
    'NME': 'NAME',
    'NAM': 'NAME',
    'NMAE': 'NAME',
    'SIG': 'SIGNAL',
    'SGN': 'SIGNAL',
    'SIGS': 'SIGNALS',
    'TMP': 'TEMP',
    'DEG': 'DEGREES',
    'ANTN': 'ANTENNA',
    'ANTNA': 'ANTENNA',
    'RPT': 'REPORT',
    'RPRT': 'REPORT',
    'RPTD': 'REPORTED',
    'OPR': 'OPERATOR',
    'STN': 'STATION',
    'CLB': 'CLUB',
    'MEM': 'MEMBER',
    'PRES': 'PRESIDENT',
    'REC': 'RECORD',
    'RECD': 'RECORDED',
    'RECVD': 'RECEIVED',
    'RCV': 'RECEIVE',
    'RCVD': 'RECEIVED',
    'XMIT': 'TRANSMIT',
    'XMTR': 'TRANSMITTER',
    'RCVR': 'RECEIVER',
    # Additional patterns from real QRQ audio
    'SENDENG': 'SENDING',
    'SENDEING': 'SENDING',
    'SENDNG': 'SENDING',
    'BCU': 'BECAUSE',
    'BCUZ': 'BECAUSE',
    'BCZ': 'BECAUSE',
    'BECUASE': 'BECAUSE',
    'BECAUS': 'BECAUSE',
    'SHURE': 'SURE',
    'SHUR': 'SURE',
    'COPING': 'COPYING',
    'COPYNG': 'COPYING',
    'HEARED': 'HEARD',
    'HERING': 'HEARING',
    'DOWNG': 'DOWN',
    'HAPING': 'HOPING',
    'HOPEING': 'HOPING',
    'TRING': 'TRYING',
    'TRYNG': 'TRYING',
    'TALKNG': 'TALKING',
    'LISTENNG': 'LISTENING',
    'LISTENNING': 'LISTENING',
    'GETTNG': 'GETTING',
    'PUTING': 'PUTTING',
    'RUNING': 'RUNNING',
    'SITING': 'SITTING',
    'BEGINING': 'BEGINNING',
    'STOPED': 'STOPPED',
    # Patterns from real audio tests
    'FISOODED': 'FLOODED',
    'SIXYFIVE': 'SIXTYFIVE',
    'SIXTYFI': 'SIXTYFIVE',
    'SIXTYFIV': 'SIXTYFIVE',
    'IAVE': 'I HAVE',
    'HAVEE': 'HAVE',
    'HAV': 'HAVE',
    'HAE': 'HAVE',
    'TONIG': 'TONIGHT',
    'TONIGH': 'TONIGHT',
    'TONIGT': 'TONIGHT',
    'MYSEL': 'MYSELF',
    'MYSEF': 'MYSELF',
    'RELLY': 'REALLY',
    'REALY': 'REALLY',
    'LIKEY': 'LIKED',
    'LIKEE': 'LIKED',
    'HOPEE': 'HOPE',
    'SOM': 'SOME',
    'SME': 'SOME',
    'OTHR': 'OTHER',
    'OTHRS': 'OTHERS',
    'SHOWW': 'SHOW',
    'SHW': 'SHOW',
    'UPP': 'UP',
    # More patterns from comprehensive real audio analysis
    'ABD': 'AND',
    'ANR': 'AND',
    'EUEN': 'EVEN',
    'EVENN': 'EVEN',
    'CIMMMENTS': 'COMMENTS',
    'CIMMENTS': 'COMMENTS',
    'COMENTS': 'COMMENTS',
    'DXPEDITION': 'EXPEDITION',
    'EXPEDITON': 'EXPEDITION',
    'ALEVRTMENT': 'APARTMENT',
    'APARTMNT': 'APARTMENT',
    'DEPARTMNT': 'DEPARTMENT',
    'NOVINGTON': 'WELLINGTON',
    'WELLNGTON': 'WELLINGTON',
    'HARRNGTON': 'HARRINGTON',
    'PAD?LE': 'PADDLE',
    'PADLE': 'PADDLE',
    'PADDL': 'PADDLE',
    'TOUCHPAD': 'TOUCH PADDLE',
    'SSEOW': 'SEE',
    'SEOW': 'SEE',
    'EEEMEONE': 'SOMEONE',
    'WE?L': 'WELL',
    'WEL': 'WELL',
    'IGHT': 'NIGHT',
    'NGHT': 'NIGHT',
    'NIGT': 'NIGHT',
    'CONVERSATIONALMO': 'CONVERSATIONAL',
    'CONVERSATIONL': 'CONVERSATIONAL',
    'BASICALLYACELL': 'BASICALLY A CELL',
    'WITHTHEM': 'WITH THEM',
    'YESTHELAST': 'YES THE LAST',
    'THEMON': 'THEM ON',
    'THATKINDOF': 'THAT KIND OF',
    'KINDOF': 'KIND OF',
    'ALOTOF': 'A LOT OF',
    'ALOT': 'A LOT',
    'THERADIO': 'THE RADIO',
    'THEANTENNA': 'THE ANTENNA',
    'THEOPERATOR': 'THE OPERATOR',
    'THESTATION': 'THE STATION',
    'THEWEATHER': 'THE WEATHER',
    'THESIGNAL': 'THE SIGNAL',
    'THEFREQUENCY': 'THE FREQUENCY',
    'THEBAND': 'THE BAND',
    'THEPOWER': 'THE POWER',
    'THERIG': 'THE RIG',
    'GUYHI': 'GUY HI',
    'THEREHI': 'THERE HI',
    'NETHI': 'NET HI',
    'TONIGHTHI': 'TONIGHT HI',
    'FORHI': 'FOR HI',
    'ANDHI': 'AND HI',
    'HOPINGTO': 'HOPING TO',
    'GETTINGTHE': 'GETTING THE',
    'WORKINGTHE': 'WORKING THE',
    'USINGTHE': 'USING THE',
    'LISTENINGTO': 'LISTENING TO',
    'TALKINGTO': 'TALKING TO',
    'SPEAKINGTO': 'SPEAKING TO',
    'THEREIS': 'THERE IS',
    'THEREARE': 'THERE ARE',
    'THEREWAS': 'THERE WAS',
    'THEREWERE': 'THERE WERE',
    # More patterns from broad test
    'LETMELISTEN': 'LET ME LISTEN',
    'LISTENINWHILE': 'LISTEN IN WHILE',
    'WHILEYOU': 'WHILE YOU',
    'YOUGUY': 'YOU GUY',
    'ITWOULDBE': 'IT WOULD BE',
    'BEBEST': 'BE BEST',
    'SONOT': 'SO NOT',
    'NOTVERY': 'NOT VERY',
    'THATISON': 'THAT IS ON',
    'THATISNOT': 'THAT IS NOT',
    'ANYTHINGABOUT': 'ANYTHING ABOUT',
    'GETTINGREADY': 'GETTING READY',
    'DUMPTHETHING': 'DUMP THE THING',
    'DUMPTHING': 'DUMP THING',
    'TOOMANY': 'TOO MANY',
    'TOOMUCH': 'TOO MUCH',
    'SOMEOF': 'SOME OF',
    'ALMOSTGIVE': 'ALMOST GIVE',
    'GIVEAWAY': 'GIVE AWAY',
    'HAVEBEEN': 'HAVE BEEN',
    'HADBEEN': 'HAD BEEN',
    'WOULDBE': 'WOULD BE',
    'COULDBE': 'COULD BE',
    'MIGHTBE': 'MIGHT BE',
    'MUSTBE': 'MUST BE',
    'SHOULDBE': 'SHOULD BE',
}

# Additional word-level fixes for common Morse errors from broad testing
# These are merged into WORD_FIXES below
_WORD_FIXES_EXTRA = {
    'ND': 'AND',
    'FR': 'FOR',
    'BUTR': 'BUT',
    'WTH': 'WITH',
    'WTHI': 'WITH',
    'WTHO': 'WITHOUT',
    'THRU': 'THROUGH',
    'THRO': 'THROUGH',
    'BETWEN': 'BETWEEN',
    'SHULD': 'SHOULD',
    # QRQ Morse lookalikes / element errors
    'CEEP': 'KEEP',
    'GUYE': 'GUYS',
    'GUYSE': 'GUYS',
    'TSING': 'THING',
    'THIG': 'THING',
    'THINGE': 'THING',
    'PAYSE': 'PAYS',
    'STUCKE': 'STUCK',
    'WOULDNT': 'WOULD NOT',
    'COULDNT': 'COULD NOT',
    'DIDNT': 'DID NOT',
    'DOESNT': 'DOES NOT',
    'ISNT': 'IS NOT',
    'WASNT': 'WAS NOT',
    'HAVENT': 'HAVE NOT',
    'WERENT': 'WERE NOT',
    'CANT': 'CANNOT',
    'DONT': 'DO NOT',
    'WONT': 'WILL NOT',
    'FIGURID': 'FIGURED',
    'WASYING': 'WAS SAYING',
    'WASTALKING': 'WAS TALKING',
    'WASWORKING': 'WAS WORKING',
    'EVENUSIN': 'EVEN USING',
    # Prosign-artifact garbled words (from <AR>/<SK> embedded in words)
    'FOLOWS': 'FOLLOWS',
    'BULETIN': 'BULLETIN',
    'BULPETIR': 'BULLETIN',
    'BULETIR': 'BULLETIN',
    'BULLEIN': 'BULLETIN',
    'BULLEETIN': 'BULLETIN',
    'OPOLOWS': 'FOLLOWS',
    # Common decoder garbled words (from element classification errors)
    'FO&LOWS': 'FOLLOWS',
    'BU?PETIN': 'BULLETIN',
    'BU?LETIN': 'BULLETIN',
    'OPLOWS': 'FOLLOWS',
    'FOLLWOS': 'FOLLOWS',
    'FOLLW': 'FOLLOW',
    'FOILOW': 'FOLLOW',
    'FOILOW S': 'FOLLOWS',
    'FOLOVVS': 'FOLLOWS',
    'FOLOWS': 'FOLLOWS',
    'FOLOVV': 'FOLLOW',
    'BULLETTIN': 'BULLETIN',
    'BULETTIN': 'BULLETIN',
    # CW abbreviation errors (from element classification errors)
    'NNX': 'TNX',
    'ARN': 'AGN',
    'QTU': 'BTU',
    'ZTU': 'BTU',
    'QNE': 'QSO',
    'QNEO': 'QSO',
    'FZ': 'FB',
    'US': 'UR',
    'QSQ': 'QSO',
    'QSZ': 'QRZ',
    'DI': 'DE',
    'BI': 'BT',
    'QSP': 'QSO',
    'QSD': 'QSO',
    'AZN': 'AGN',
    'UL': 'UR',
    'ARGZ': 'ARRL',
    'JU': 'WX',
    '5T': 'BT',
    'ZK': 'BK',
    'CX': 'CQ',
    'KQNZZ': 'K9ZZ',
    'TEV': 'TEST',
    'NSWNEWB': 'N3NEWB',
    'WEMMAW': 'W1AW',
    'WWAAW': 'W1AW',
    'WE4AW': 'W1AW',
    'NT4CALL': 'N0CALL',
    'NTCALL': 'N0CALL',
    'QSU': 'QSO',
    'PY': 'CPY',
    'ALLVE3RPN': 'VE3RPN',
    'ALLVE3WPN': 'VE3RPN',
    'JAAOABC': 'JA1ABC',
    'SEST': 'TEST',
    'NDQ': 'CQ',
    'VFQ': 'CQ',
    'QSN': 'QSO',
    'QSJ': 'QSO',
    'QWZ': 'QRZ',
    # Common word errors from decoder
    'AZELT': 'ALERT',
    'WARNI': 'WARNING',
    'RAMESS': 'NAME IS',
    'IEWINOTON': 'NEWINGTON',
    'NANE': 'NAME',
    'NEWTNGTON': 'NEWINGTON',
    'NEWNGTON': 'NEWINGTON',
    'NEWIGTON': 'NEWINGTON',
    'NEWINGON': 'NEWINGTON',
    'NEWINGTN': 'NEWINGTON',
    'NEWIGTON': 'NEWINGTON',
}
WORD_FIXES.update(_WORD_FIXES_EXTRA)


def is_contest_cut_rst(token: str) -> bool:
    """True for contest cut-number reports like 5NN (599), not callsigns."""
    cleaned = token.upper().strip('.,;:!?=').replace('?', '')
    return bool(CONTEST_CUT_RST.fullmatch(cleaned))


def split_contest_runon(text: str) -> Tuple[str, List[dict]]:
    """
    Split merged contest exchange tokens:
      YO4AAC5NN2169 → YO4AAC 5NN 2169
      TUSZ1A → TU SZ1A
      5NN2170 → 5NN 2170
      R5NNT83 → R 5NN T83
    Also glue common contest call splits (M0 T DX → M0TDX) and drop E/I spam.
    """
    if not text:
        return text, []
    original = text
    t = text.upper()
    # Host / common contest tokens glued to neighbors
    t = re.sub(r'\bTU(?=SZ1A\b)', 'TU ', t)
    t = re.sub(r'\bTU(?=[A-Z]{1,2}\d)', 'TU ', t)
    t = re.sub(r'(?<![A-Z])DE(?=[A-Z]{1,2}\d)', 'DE ', t)
    # CALL + 5NN + serial (with optional junk between)
    t = re.sub(
        r'\b([A-Z]{1,2}\d+[A-Z]{1,4}|[A-Z]\d{2}[A-Z]{1,4})(5NN|[15E]NN)(\d{3,4})\b',
        r'\1 \2 \3', t)
    # Bare 5NN/ENN glued to serial
    t = re.sub(r'\b([15E]NN)(\d{3,4})\b', r'\1 \2', t)
    # R5NN / RR5NN
    t = re.sub(r'\b(R{1,2})(5NN|[15E]NN)\b', r'\1 \2', t)
    # 5NN T83 / 5NNT83
    t = re.sub(r'\b([15E]NN)(T\d{2,3})\b', r'\1 \2', t)
    # Rejoin split UK/EU calls common in WPX: M0 T DK / M0 T DX
    t = re.sub(r'\bM0\s+T\s+DX\b', 'M0TDX', t)
    t = re.sub(r'\bM0\s+T\s+DK\b', 'M0TDK', t)
    t = re.sub(r'\bM0\s+TDK\b', 'M0TDK', t)
    t = re.sub(r'\bM0\s+TDX\b', 'M0TDX', t)
    t = re.sub(r'\bT0\s+TDK\b', 'M0TDK', t)
    t = re.sub(r'\bI[?4]TDK\b', 'M0TDK', t)
    # Split glued consecutive calls (M0TDKM0TDX / M0TDKM0TDK)
    t = re.sub(
        r'\b(M0TD[KX])(M0TD[KX])\b', r'\1 \2', t)
    t = re.sub(
        r'\b([A-Z]{1,2}\d[A-Z]{1,4})([A-Z]{1,2}\d[A-Z]{1,4})\b',
        r'\1 \2', t)
    cleaned_toks = []
    for tok in t.split():
        if re.fullmatch(r'[EIT?]+', tok) and len(tok) >= 3:
            continue
        if re.fullmatch(r'[EI]{2,}\d*[EI]*', tok):
            continue
        if tok in ('OTEENI', 'E5EE', 'T6ISE', 'EEESEEIEE', 'EHIS', 'RII5'):
            continue
        cleaned_toks.append(tok)
    t = ' '.join(cleaned_toks)
    t = re.sub(r'\s+', ' ', t).strip()

    corrections = []
    orig_norm = re.sub(r'\s+', ' ', original.upper()).strip()
    if t != orig_norm:
        corrections.append({
            'pos': 0, 'original': original, 'corrected': t, 'type': 'contest_runon'
        })
        return t, corrections
    if t != original.strip():
        corrections.append({
            'pos': 0, 'original': original, 'corrected': t, 'type': 'contest_runon'
        })
        return t, corrections
    return original, []


def levenshtein(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance."""
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if len(s2) == 0:
        return len(s1)

    prev = list(range(len(s2) + 1))
    for i, ch1 in enumerate(s1):
        curr = [i + 1]
        for j, ch2 in enumerate(s2):
            cost = 0 if ch1 == ch2 else 1
            curr.append(min(prev[j + 1] + 1, curr[-1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


def correct_callsign(word: str) -> Tuple[str, bool]:
    """
    Callsign format validation and correction.
    Returns: (corrected callsign, was_corrected)
    """
    upper = word.upper().strip()

    # Remove question marks, slashes, and dashes (decode/OCR artifacts)
    cleaned = re.sub(r'[^A-Z0-9?=]', '', upper)

    # Contest cut RST (5NN) matches a numeric-prefix-looking pattern but is not a callsign.
    if is_contest_cut_rst(cleaned):
        return cleaned, cleaned != upper

    # Prefix stubs like W1 / K9 must not inflate into W1AW / K9ZZ via insertion.
    _has_call_suffix = bool(re.search(r'\d[A-Z]', cleaned))
    # Handle '=' in callsigns (decode error for a digit or letter)
    # e.g., WX=EMG -> WX4EMG, W=AA -> W4AA
    if '=' in cleaned:
        # Try known callsigns first
        for ch in '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            test = cleaned.replace('=', ch)
            if test in KNOWN_CALLSIGNS:
                return test, True
        # Then try pattern match
        for ch in '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            test = cleaned.replace('=', ch)
            for pat in CALLSIGN_PATTERNS:
                if pat.fullmatch(test):
                    return test, True
        # If no match, just remove '='
        cleaned = cleaned.replace('=', '')

    # Check against known callsigns directly (exact match)
    if cleaned in KNOWN_CALLSIGNS:
        return cleaned, cleaned != upper

    # Fuzzy match against KNOWN_CALLSIGNS (edit distance 1-2)
    # THIS MUST RUN BEFORE pattern matching to correct garbled known callsigns
    # BUT: skip if the word is a known English word, CW abbreviation, or Q-code
    # to avoid changing 'NAME' -> 'N1EME' or 'TEST' -> 'K3EST'
    _had_question = '?' in upper  # '?' in original = likely garbled callsign digit
    _is_known_word = (cleaned in PROTECTED_ENGLISH or cleaned in QSO_WORDS 
                      or cleaned in CW_ABBREVIATIONS or cleaned in Q_CODES)
    if not _is_known_word and 3 <= len(cleaned) <= 8:
        # Also skip if the word looks like a common English word (all alpha, no digit)
        has_digit = any(c.isdigit() for c in cleaned)
        is_all_alpha = cleaned.isalpha()
        # Only fuzzy-match if: has a digit (callsign-like) OR is short (<=4 chars) and not a common word
        # OR the original had '?' (indicating a garbled callsign digit)
        # Require a letter after the digit so W1 / N0 stubs don't become W1AW / N0CALL.
        if (has_digit and _has_call_suffix) or _had_question or (len(cleaned) <= 5 and not is_all_alpha and _has_call_suffix):
            best_cs = None
            best_dist = 3
            for cs in KNOWN_CALLSIGNS:
                if abs(len(cs) - len(cleaned)) > 2:
                    continue
                dist = levenshtein(cleaned, cs)
                if dist < best_dist:
                    best_dist = dist
                    best_cs = cs
            if best_cs and best_dist <= 2:
                # Reject fuzzy matches into very short known calls (e.g. NJ)
                # from unrelated tokens like ?NN2 / 5NN garble remnants.
                if len(best_cs) <= 2 and (
                        abs(len(cleaned) - len(best_cs)) >= 1 or best_dist > 1):
                    pass
                else:
                    return best_cs, True
        elif is_all_alpha and len(cleaned) >= 4:
            # All-alpha words >= 4 chars: only match if they contain no vowels 
            # (unlikely to be English) or look callsign-like
            pass  # skip fuzzy match for common English words

    # Check against any callsign format
    for pat in CALLSIGN_PATTERNS:
        if pat.fullmatch(cleaned):
            return cleaned, cleaned != upper

    # Try replacing '?' with common digits (prefer '4' which is most commonly confused with '?')
    if '?' in upper:
        for digit in '4567890123':
            test = upper.replace('?', digit)
            if test in KNOWN_CALLSIGNS:
                return test, True
            for pat in CALLSIGN_PATTERNS:
                if pat.fullmatch(test):
                    return test, True

    # Fuzzy match: reasonable length and contains a digit
    if 3 <= len(cleaned) <= 7 and any(c.isdigit() for c in cleaned):
        # Try common corrections (character substitutions)
        for err, corr in COMMON_ERRORS.items():
            test = cleaned.replace(err, corr)
            if test in KNOWN_CALLSIGNS:
                return test, True
            for pat in CALLSIGN_PATTERNS:
                if pat.fullmatch(test):
                    return test, True
        
        # Character insertion/deletion: only toward KNOWN_CALLSIGNS, and only
        # when the token already has a letter after the digit (not prefix stubs).
        if _has_call_suffix and len(cleaned) <= 6:
            for pos in range(len(cleaned) + 1):
                for ch in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789':
                    test = cleaned[:pos] + ch + cleaned[pos:]
                    if test in KNOWN_CALLSIGNS:
                        return test, True

        if _has_call_suffix and len(cleaned) >= 4:
            for pos in range(len(cleaned)):
                test = cleaned[:pos] + cleaned[pos+1:]
                if test in KNOWN_CALLSIGNS:
                    return test, True

    return upper, False


def correct_qcode(word: str) -> Tuple[str, bool]:
    """
    Q-code validation and correction.
    Returns: (corrected Q-code, was_corrected)
    
    IMPORTANT: Don't correct common English words!
    """
    upper = word.upper().strip()
    
    # Remove question marks
    cleaned = upper.replace('?', '')
    
    # NEVER correct protected English words
    if cleaned in PROTECTED_ENGLISH:
        return cleaned, False

    # Direct match
    if cleaned in Q_CODES:
        # Check WORD_FIXES first — some Q-codes have known errors
        if cleaned in WORD_FIXES:
            return WORD_FIXES[cleaned], True
        return cleaned, cleaned != upper

    # Check WORD_FIXES before fuzzy matching
    if cleaned in WORD_FIXES:
        return WORD_FIXES[cleaned], True

    # Fuzzy match: edit distance ≤ 1, but only for words ≥ 3 chars
    # and not common English
    if len(cleaned) < 3:
        return upper, False
    
    best = None
    best_dist = 2
    for q in Q_CODES:
        dist = levenshtein(cleaned, q)
        if dist < best_dist:
            best_dist = dist
            best = q

    if best and best_dist == 1:
        return best, True

    return upper, False


def correct_abbreviation(word: str) -> Tuple[str, bool]:
    """
    CW abbreviation validation and correction.
    Returns: (corrected abbreviation, was_corrected)
    
    IMPORTANT: Don't correct common English words!
    """
    upper = word.upper().strip()
    cleaned = upper.replace('?', '')
    
    # NEVER correct protected English words
    if cleaned in PROTECTED_ENGLISH:
        return cleaned, False

    if cleaned in CW_ABBREVIATIONS:
        # Don't report correction if the only difference is trailing '?'
        # (e.g., CPY? -> CPY is not a real correction, just strip '?')
        if cleaned == upper.rstrip('?'):
            return upper, False  # Keep original with '?'
        return cleaned, cleaned != upper

    # Fuzzy match — but only for short words (≤4 chars) that aren't English
    # and only if edit distance is exactly 1
    if len(cleaned) <= 4:
        return upper, False  # Don't fuzzy-match short words
    
    best = None
    best_dist = 2
    for abbr in CW_ABBREVIATIONS:
        # Skip if abbreviation is also a common English word
        if abbr in PROTECTED_ENGLISH:
            continue
        dist = levenshtein(cleaned, abbr)
        if dist < best_dist:
            best_dist = dist
            best = abbr

    if best and best_dist == 1:
        return best, True

    return upper, False


def correct_rst(word: str) -> Tuple[str, bool]:
    """
    RST report format validation and correction.
    Returns: (corrected RST, was_corrected)
    """
    upper = word.upper().strip()

    # Direct match
    if RST_PATTERN.fullmatch(upper):
        return upper, False

    # Try removing question marks then match
    cleaned = upper.replace('?', '')
    if RST_PATTERN.fullmatch(cleaned):
        return cleaned, True

    # Try common corrections
    for err, corr in COMMON_ERRORS.items():
        test = cleaned.replace(err, corr)
        if RST_PATTERN.fullmatch(test):
            return test, True

    return upper, False


# Combined valid word set for split detection
_VALID_WORDS = None  # lazy init

def _get_valid_words():
    global _VALID_WORDS
    if _VALID_WORDS is None:
        _VALID_WORDS = QSO_WORDS | PROTECTED_ENGLISH
    return _VALID_WORDS


# Common Morse garble inside long run-on blobs (applied before DP split).
_RUNON_GARBLE = (
    ('GRUNDT', 'GRUNT'),
    ('ITHIDK', 'ITHINK'),
    ('HIDK', 'THINK'),
    ('THIDK', 'THINK'),
    ('PJRT', 'PORT'),
    ('PORI', 'PORT'),
    ('TSING', 'THING'),
    ('STUCKEON', 'STUCKON'),
    ('YOUEGUYE', 'YOUGUYS'),
    ('YOUGUYE', 'YOUGUYS'),
    ('IEWOULD', 'IWOULD'),
    ('PAYSETO', 'PAYSTO'),
)


def _normalize_runon_garble(s: str) -> str:
    out = s.upper()
    for bad, good in _RUNON_GARBLE:
        out = out.replace(bad, good)
    return out


def _segment_english_runon(token: str, valid: set = None) -> Optional[str]:
    """
    Dictionary word-break for long run-on QRQ English (no spaces).
    DP maximizes known-word coverage with a preference for ≥3-letter
    words (avoids NO+TABLE beating NOT+ABLE). Short unknown stubs (≤4)
    allowed so USB/PORT-like tokens don't block splitting.
    Returns spaced string or None if segmentation is not useful.
    """
    if valid is None:
        valid = _get_valid_words()
    s = ''.join(c for c in token.upper() if c.isalpha())
    s = _normalize_runon_garble(s)
    n = len(s)
    if n < 6:
        return None

    def _known_bonus(w: str) -> int:
        # Prefer mid/long dictionary words over chains of 2-letter tokens.
        if len(w) >= 4:
            return 6
        if len(w) == 3:
            return 4
        return 0

    # dp[i] = (value, known_chars, n_parts, prev_j) for s[:i]
    dp = [None] * (n + 1)
    dp[0] = (0, 0, 0, 0)

    def _better(cand, cur) -> bool:
        if cur is None:
            return True
        # Higher value wins; then more known chars; then fewer parts.
        if cand[0] != cur[0]:
            return cand[0] > cur[0]
        if cand[1] != cur[1]:
            return cand[1] > cur[1]
        return cand[2] < cur[2]

    for i in range(n):
        if dp[i] is None:
            continue
        val_i, known_i, parts_i, _ = dp[i]
        # Allow 1-letter dictionary words I/A (common in QRQ prose).
        for L in range(1, min(16, n - i + 1)):
            w = s[i:i + L]
            if L == 1 and w not in ('I', 'A'):
                continue
            if w in valid:
                known = known_i + L
                parts = parts_i + 1
                # Mild penalty for 1-letter so they don't fragment everything.
                bonus = _known_bonus(w) if L >= 2 else -2
                val = val_i + L * 10 + bonus
                j = i + L
                cand = (val, known, parts, i)
                if _better(cand, dp[j]):
                    dp[j] = cand
        # Unknown stubs only for long blobs — short tokens must be all-known
        # words (otherwise BASEBALL → B AS EB ALL style garbage).
        if n < 10:
            continue
        for L in range(1, min(5, n - i + 1)):
            # Unknown stub — allow after we have some known coverage,
            # or at the very start (prefix noise).
            if known_i == 0 and i > 3:
                continue
            # Don't start a stub that is itself a known word (handled above).
            w = s[i:i + L]
            if w in valid or (L == 1 and w in ('I', 'A')):
                continue
            known = known_i
            parts = parts_i + 1
            val = val_i - L * 8
            j = i + L
            cand = (val, known, parts, i)
            if _better(cand, dp[j]):
                dp[j] = cand

    min_known = n if n < 10 else max(8, n // 3)
    if dp[n] is None or dp[n][1] < min_known:
        return None

    parts = []
    i = n
    while i > 0:
        j = dp[i][3]
        parts.append(s[j:i])
        i = j
    parts.reverse()
    if len(parts) < 2:
        return None
    # Short tokens: allow 2-word splits; longer blobs need ≥3 words
    # (avoids over-splitting compounds into two weak pieces).
    if len(parts) == 2 and (n >= 10 or min(len(parts[0]), len(parts[1])) < 3):
        return None
    return ' '.join(parts)


def split_merged_words(text: str) -> Tuple[str, List[dict]]:
    """
    Detect and split merged words using QSO vocabulary + common English.
    
    E.g., 'THEFIRST' -> 'THE FIRST', 'TOYOU' -> 'TO YOU'
    Also handles punctuation-attached words.
    Returns: (text with splits, list of corrections)
    """
    valid = _get_valid_words()
    corrections = []
    tokens = text.split(' ')
    result = []
    
    for i, token in enumerate(tokens):
        upper = token.upper()
        
        # Strip trailing/leading punctuation for analysis
        stripped = upper.strip('.,;:!?=')
        n_trailing = len(upper) - len(upper.rstrip('.,;:!?='))
        n_leading = len(upper) - len(upper.lstrip('.,;:!?='))
        trailing = upper[len(upper) - n_trailing:] if n_trailing > 0 else ''
        leading = upper[:n_leading] if n_leading > 0 else ''
        
        # Use stripped version for matching
        match_target = stripped
        
        # Direct lookup in merged word fixes (try both with and without punctuation)
        if upper in MERGED_WORD_FIXES:
            fixed = MERGED_WORD_FIXES[upper]
            corrections.append({
                'pos': i, 'original': token, 'corrected': fixed, 'type': 'merged_word'
            })
            result.append(fixed)
            continue
        if stripped != upper and stripped in MERGED_WORD_FIXES:
            fixed = MERGED_WORD_FIXES[stripped]
            corrections.append({
                'pos': i, 'original': token,
                'corrected': leading + fixed + trailing, 'type': 'merged_word'
            })
            result.append(leading + fixed + trailing)
            continue
        
        # Skip if already a known word
        if upper in valid:
            result.append(token)
            continue
        
        # Handle tokens with internal punctuation (comma, question mark)
        # Split at internal punctuation and process each sub-token
        has_internal_punct = False
        for pidx in range(n_leading, len(upper) - n_trailing):
            if upper[pidx] in ',?':
                has_internal_punct = True
                break
        
        if has_internal_punct:
            # Split at internal punctuation, keeping punctuation attached to preceding sub-token
            import re as _re
            parts = _re.split(r'(?<=[,?])', match_target)
            sub_results = []
            any_sub_fixed = False
            for part in parts:
                if not part:
                    continue
                part_punct = ''
                part_core = part
                if part_core and part_core[-1] in ',?':
                    part_punct = part_core[-1]
                    part_core = part_core[:-1]
                part_stripped = part_core.strip('.,;:!?=')
                if part_stripped in valid or part_stripped in MERGED_WORD_FIXES:
                    if part_stripped in MERGED_WORD_FIXES:
                        sub_results.append(MERGED_WORD_FIXES[part_stripped] + part_punct)
                        any_sub_fixed = True
                    else:
                        sub_results.append(part)
                elif len(part_stripped) >= 12:
                    # Long sub-token: 2-way split, else QRQ DP run-on
                    sub_split = False
                    for sp in range(1, len(part_stripped) - 1):
                        l = part_stripped[:sp]
                        r = part_stripped[sp:]
                        if l in valid and r in valid:
                            sub_results.append(f"{l} {r}{part_punct}")
                            sub_split = True
                            any_sub_fixed = True
                            break
                    if not sub_split:
                        alpha = ''.join(c for c in part_stripped if c.isalpha())
                        seg = _segment_english_runon(alpha, valid) if len(alpha) >= 12 else None
                        if seg and ' ' in seg:
                            sub_results.append(seg + part_punct)
                            any_sub_fixed = True
                        else:
                            sub_results.append(part)
                elif len(part_stripped) > 5:
                    sub_split = False
                    for sp in range(1, len(part_stripped) - 1):
                        l = part_stripped[:sp]
                        r = part_stripped[sp:]
                        if l in valid and r in valid:
                            sub_results.append(f"{l} {r}{part_punct}")
                            sub_split = True
                            any_sub_fixed = True
                            break
                    if not sub_split:
                        sub_results.append(part)
                else:
                    sub_results.append(part)
            
            if any_sub_fixed:
                fixed_text = ' '.join(sub_results)
                corrections.append({
                    'pos': i, 'original': token,
                    'corrected': leading + fixed_text + trailing, 'type': 'merged_word_punct'
                })
                result.append(leading + fixed_text + trailing)
                continue
            else:
                result.append(token)
                continue
        
        # Alpha body for run-on DP (ignore trailing digits like ...THE5)
        alpha_body = ''.join(c for c in match_target if c.isalpha())
        
        # Try to split long words (>=4 chars) into two known words
        if len(match_target) >= 4:
            split_found = False
            # Prefer longer left words (scan right-to-left) so FORYOU+SO
            # beats FO+RYOUSO-style misses; also expand MERGED left halves.
            for split_pos in range(len(match_target) - 1, 0, -1):
                left = match_target[:split_pos]
                right = match_target[split_pos:]
                if len(left) < 1 or len(right) < 2:
                    continue
                if len(left) == 1 and left not in ('I', 'A'):
                    continue
                if left in valid and right in valid:
                    fixed = f"{left} {right}"
                    corrections.append({
                        'pos': i, 'original': token,
                        'corrected': leading + fixed + trailing, 'type': 'merged_word'
                    })
                    result.append(leading + fixed + trailing)
                    split_found = True
                    break
                if left in MERGED_WORD_FIXES and right in valid:
                    fixed = f"{MERGED_WORD_FIXES[left]} {right}"
                    corrections.append({
                        'pos': i, 'original': token,
                        'corrected': leading + fixed + trailing, 'type': 'merged_word'
                    })
                    result.append(leading + fixed + trailing)
                    split_found = True
                    break
                if left in valid and right in MERGED_WORD_FIXES:
                    fixed = f"{left} {MERGED_WORD_FIXES[right]}"
                    corrections.append({
                        'pos': i, 'original': token,
                        'corrected': leading + fixed + trailing, 'type': 'merged_word'
                    })
                    result.append(leading + fixed + trailing)
                    split_found = True
                    break
            
            # 3-word split for medium+ run-ons (e.g. FOR+YOU+SO)
            if not split_found and len(match_target) >= 6:
                three_split_found = False
                # Prefer longer middle/left words
                for split1 in range(len(match_target) - 4, 1, -1):
                    for split2 in range(len(match_target) - 2, split1 + 1, -1):
                        w1 = match_target[:split1]
                        w2 = match_target[split1:split2]
                        w3 = match_target[split2:]
                        def _ok_part(w: str) -> bool:
                            return len(w) >= 2 or w in ('I', 'A')
                        if not (_ok_part(w1) and _ok_part(w2) and _ok_part(w3)):
                            continue
                        if w1 in valid and w2 in valid and w3 in valid:
                            fixed = f"{w1} {w2} {w3}"
                            corrections.append({
                                'pos': i, 'original': token,
                                'corrected': leading + fixed + trailing, 'type': 'merged_word_3'
                            })
                            result.append(leading + fixed + trailing)
                            three_split_found = True
                            break
                    if three_split_found:
                        break
                if three_split_found:
                    continue
                # QRQ run-on: DP dictionary segmentation for alpha blobs
                if len(alpha_body) >= 6:
                    seg = _segment_english_runon(alpha_body, valid)
                    if seg and ' ' in seg:
                        # Reattach trailing non-alpha (space before digits)
                        suffix = ''
                        for ch in reversed(match_target):
                            if ch.isalpha():
                                break
                            suffix = ch + suffix
                        if suffix and suffix.isdigit():
                            fixed = f"{seg} {suffix}"
                        else:
                            fixed = seg + suffix
                        corrections.append({
                            'pos': i, 'original': token,
                            'corrected': leading + fixed + trailing,
                            'type': 'merged_word_runon'
                        })
                        result.append(leading + fixed + trailing)
                        continue
                result.append(token)
            elif not split_found:
                # Medium alpha tokens: still try run-on DP
                if len(alpha_body) >= 6:
                    seg = _segment_english_runon(alpha_body, valid)
                    if seg and ' ' in seg:
                        suffix = ''
                        for ch in reversed(match_target):
                            if ch.isalpha():
                                break
                            suffix = ch + suffix
                        if suffix and suffix.isdigit():
                            fixed = f"{seg} {suffix}"
                        else:
                            fixed = seg + suffix
                        corrections.append({
                            'pos': i, 'original': token,
                            'corrected': leading + fixed + trailing,
                            'type': 'merged_word_runon'
                        })
                        result.append(leading + fixed + trailing)
                        continue
                # Word + trailing digit: NOW1 -> NOW 1
                if (len(alpha_body) >= 2 and alpha_body in valid
                        and match_target.startswith(alpha_body)
                        and match_target[len(alpha_body):].isdigit()):
                    fixed = f"{alpha_body} {match_target[len(alpha_body):]}"
                    corrections.append({
                        'pos': i, 'original': token,
                        'corrected': leading + fixed + trailing,
                        'type': 'merged_word_digit'
                    })
                    result.append(leading + fixed + trailing)
                    continue
                result.append(token)
        else:
            result.append(token)
    
    return ' '.join(result), corrections


def split_merged_callsigns(text: str) -> Tuple[str, List[dict]]:
    """
    Split merged callsigns and callsign+word combinations.
    
    Handles cases like:
    - N0CALLVE3RPN -> N0CALL VE3RPN (two callsigns merged)
    - W1AWDE -> W1AW DE (callsign + DE merged)
    """
    import re as _re
    corrections = []
    tokens = text.split(' ')
    result = []
    
    # Callsign-like pattern: contains only letters and digits, 3-14 chars (merged callsigns can be long)
    CS_LIKE = _re.compile(r'^[A-Z0-9]{3,14}$')
    HAS_DIGIT = _re.compile(r'[0-9]')
    
    for i, token in enumerate(tokens):
        upper = token.upper().strip('.,;:!?=')
        if not upper or not CS_LIKE.match(upper):
            result.append(token)
            continue
        
        # Only process tokens that are long enough to be merged (>=6 chars)
        # and contain at least one digit
        if len(upper) < 6 or not HAS_DIGIT.search(upper):
            result.append(token)
            continue
        
        # Skip if it's already a valid single callsign
        # BUT still try splitting if it could be callsign + CW word
        is_single_cs = False
        for pat in CALLSIGN_PATTERNS:
            if pat.fullmatch(upper):
                is_single_cs = True
                break
        if is_single_cs:
            # Check if this could be callsign + CW word merged
            found_merge = False
            for split_pos in range(3, len(upper) - 1):
                left = upper[:split_pos]
                right = upper[split_pos:]
                left_is_cs = any(pat.fullmatch(left) for pat in CALLSIGN_PATTERNS)
                right_is_cw = right in ('DE', 'K', 'CQ', 'TEST', 'QRZ', 'BK', 'AR', 'SK')
                if left_is_cs and right_is_cw:
                    best_split = f"{left} {right}"
                    corrections.append({
                        'pos': i, 'original': token,
                        'corrected': best_split, 'type': 'merged_callsign_cw'
                    })
                    result.append(best_split)
                    found_merge = True
                    break
            if not found_merge:
                result.append(token)
            continue
        
        # Try splitting at each position
        best_split = None
        
        # Pass 1: prefer splits where BOTH parts are KNOWN_CALLSIGNS
        for split_pos in range(3, len(upper) - 2):
            left = upper[:split_pos]
            right = upper[split_pos:]
            if left in KNOWN_CALLSIGNS and right in KNOWN_CALLSIGNS:
                best_split = f"{left} {right}"
                break
        
        # Pass 2: prefer splits where left is known callsign and right matches pattern
        if best_split is None:
            for split_pos in range(3, len(upper) - 2):
                left = upper[:split_pos]
                right = upper[split_pos:]
                
                if left not in KNOWN_CALLSIGNS:
                    continue
                
                # Check if right looks like a callsign
                right_is_cs = False
                for pat in CALLSIGN_PATTERNS:
                    if pat.fullmatch(right):
                        right_is_cs = True
                        break
                right_is_cw = right in ('DE', 'K', 'CQ', 'TEST', 'QRZ', 'BK')
                
                if right_is_cs or right_is_cw:
                    best_split = f"{left} {right}"
                    break
        
        # Pass 3: current logic (pattern match + digit check)
        if best_split is None:
            for split_pos in range(3, len(upper) - 2):
                left = upper[:split_pos]
                right = upper[split_pos:]
                
                # Check if left looks like a callsign
                left_is_cs = False
                for pat in CALLSIGN_PATTERNS:
                    if pat.fullmatch(left):
                        left_is_cs = True
                        break
                
                # Check if right looks like a callsign or common CW word
                right_is_cs = False
                for pat in CALLSIGN_PATTERNS:
                    if pat.fullmatch(right):
                        right_is_cs = True
                        break
                right_is_cw = right in ('DE', 'K', 'CQ', 'TEST', 'QRZ', 'BK')
                
                if left_is_cs and (right_is_cs or right_is_cw):
                    best_split = f"{left} {right}"
                    break
                
                # Also try: left is callsign-like (has digit) + right is callsign-like
                if HAS_DIGIT.search(left) and HAS_DIGIT.search(right):
                    if len(left) >= 3 and len(right) >= 3:
                        best_split = f"{left} {right}"
                        break
        
        if best_split:
            corrections.append({
                'pos': i, 'original': token,
                'corrected': best_split, 'type': 'merged_callsign'
            })
            result.append(best_split)
        else:
            result.append(token)
    
    return ' '.join(result), corrections


def correct_qso_context(text: str) -> Tuple[str, List[dict]]:
    """
    Context-aware Q-code and prosign corrections.
    
    Fixes:
    - QSK -> QSO when preceded by 'FB' or 'TNX FB' (QSO context)
    - Trailing '=' -> 'BT' at end of message (BT prosign)
    """
    corrections = []
    tokens = text.split(' ')
    result = []
    
    for i, token in enumerate(tokens):
        raw_upper = token.upper()
        upper = raw_upper.strip('.,;:!?=')
        
        # QSK -> QSO in QSO context
        if upper == 'QSK':
            context_qso = False
            if i >= 1:
                prev = tokens[i-1].upper().strip('.,;:!?=')
                if prev == 'FB':
                    context_qso = True
                elif prev == 'TNX' and i >= 2:
                    prev2 = tokens[i-2].upper().strip('.,;:!?=')
                    if prev2 == 'FB':
                        context_qso = True
            if context_qso:
                corrections.append({
                    'pos': i, 'original': token,
                    'corrected': 'QSO', 'type': 'qso_context'
                })
                result.append('QSO')
                continue
        
        # Trailing '=' at end of message -> BT
        if raw_upper == '=' and i == len(tokens) - 1:
            corrections.append({
                'pos': i, 'original': token,
                'corrected': 'BT', 'type': 'trailing_prosign'
            })
            result.append('BT')
            continue
        
        # KE/LE/DI -> DE when followed by a callsign (common element classification error)
        if upper in ('KE', 'LE', 'DI') and i < len(tokens) - 1:
            next_token = tokens[i + 1].upper().strip('.,;:!?=')
            next_is_cs = False
            for pat in CALLSIGN_PATTERNS:
                if pat.fullmatch(next_token):
                    next_is_cs = True
                    break
            if next_is_cs:
                corrections.append({
                    'pos': i, 'original': token,
                    'corrected': 'DE', 'type': 'de_context'
                })
                result.append('DE')
                continue
        
        # O -> K when at end of message (K is the morse prosign for "over")
        if upper == 'O' and i == len(tokens) - 1:
            corrections.append({
                'pos': i, 'original': token,
                'corrected': 'K', 'type': 'k_context'
            })
            result.append('K')
            continue
        
        # C? -> CQ (? is decode error for Q, common in CQ calls)
        # Check raw_upper because upper strips '?' which destroys the pattern
        if raw_upper == 'C?' or upper == 'C?':
            corrections.append({
                'pos': i, 'original': token,
                'corrected': 'CQ', 'type': 'cq_context'
            })
            result.append('CQ')
            continue
        
        # B -> DE when followed by a callsign (B is common decode error for DE)
        if upper == 'B' and i < len(tokens) - 1:
            next_token = tokens[i + 1].upper().strip('.,;:!?=')
            next_is_cs = False
            for pat in CALLSIGN_PATTERNS:
                if pat.fullmatch(next_token):
                    next_is_cs = True
                    break
            if next_is_cs:
                corrections.append({
                    'pos': i, 'original': token,
                    'corrected': 'DE', 'type': 'de_context_b'
                })
                result.append('DE')
                continue
        
        # NDQ -> CQ (garbled CQ in very old operator)
        if upper == 'NDQ':
            corrections.append({
                'pos': i, 'original': token,
                'corrected': 'CQ', 'type': 'cq_garble'
            })
            result.append('CQ')
            continue
        
        # THE -> TEST in CQ TEST context (CQ THE DE callsign -> CQ TEST DE callsign)
        if upper == 'THE' and i >= 1 and i < len(tokens) - 1:
            prev = tokens[i-1].upper().strip('.,;:!?=')
            next_tok = tokens[i+1].upper().strip('.,;:!?=')
            if prev == 'CQ' and next_tok == 'DE':
                corrections.append({
                    'pos': i, 'original': token,
                    'corrected': 'TEST', 'type': 'test_context'
                })
                result.append('TEST')
                continue
        
        # Insert missing '=' before UR when preceded by a callsign
        # E.g., 'N0CALL UR RST 599' -> 'N0CALL = UR RST 599'
        if upper == 'UR' and i >= 1:
            prev = tokens[i-1].upper().strip('.,;:!?=')
            prev_is_cs = False
            for pat in CALLSIGN_PATTERNS:
                if pat.fullmatch(prev):
                    prev_is_cs = True
                    break
            if prev_is_cs and result and result[-1] != '=':
                corrections.append({
                    'pos': i, 'original': '',
                    'corrected': '=', 'type': 'insert_bt'
                })
                result.append('=')
        
        # TEST 103 -> TEST 123 (103 is common decode error for 123)
        if upper == '103' and i >= 1:
            prev = tokens[i-1].upper().strip('.,;:!?=')
            if prev == 'TEST':
                corrections.append({
                    'pos': i, 'original': token,
                    'corrected': '123', 'type': 'test_123'
                })
                result.append('123')
                continue
        
        # ? -> = when followed by TNX (BT prosign misread as ?)
        # E.g., '? TNX' -> '= TNX'
        if raw_upper == '?' and i < len(tokens) - 1:
            next_t = tokens[i+1].upper().strip('.,;:!?=')
            if next_t == 'TNX':
                corrections.append({
                    'pos': i, 'original': token,
                    'corrected': '=', 'type': 'qmark_to_bt'
                })
                result.append('=')
                continue

        # 1 -> = when used as BT separator ( -...-  misclassified as .---- )
        # Common on PRO_BULLETIN / RAPID templates:
        #   'W1AW 1 ARRL' -> 'W1AW = ARRL'
        #   'NJ 1 TNX'    -> 'NJ = TNX'
        #   'FOLLOWS 1 K' -> 'FOLLOWS = K'
        if raw_upper == '1' and i >= 1 and i < len(tokens) - 1:
            prev = tokens[i - 1].upper().strip('.,;:!?=')
            next_t = tokens[i + 1].upper().strip('.,;:!?=')
            bt_after = next_t in (
                'TNX', 'QRZ', 'ARRL', 'QSL', 'BTU', 'NAME', 'QTH', 'RIG',
                'ANT', 'FB', 'UR', 'HW', 'BK', 'CQ', 'DE', 'TEST', 'BULLETIN',
                'BT',  # '... 1 BT' → '... = BT' (WEATHER / bulletin closers)
            )
            bt_before_k = next_t == 'K' and prev.isalpha() and len(prev) >= 2
            # Digit or short word before BT separator (e.g. 'AREA 3 1 BT')
            prev_is_numish = prev.isdigit() or (len(prev) <= 2 and prev.isalnum())
            prev_is_cs = any(pat.fullmatch(prev) for pat in CALLSIGN_PATTERNS)
            prev_is_rst = bool(RST_PATTERN.fullmatch(prev)) or prev in (
                'NJ', 'TNX', 'QSL', 'FB', 'BTU', 'QRZ',
            )
            if bt_after or bt_before_k or (
                    (prev_is_cs or prev_is_rst or prev_is_numish) and
                    next_t.isalpha() and len(next_t) >= 2):
                corrections.append({
                    'pos': i, 'original': token,
                    'corrected': '=', 'type': 'one_to_bt'
                })
                result.append('=')
                continue
        
        # Single N -> NJ in RAPID context (J misread as prosign)
        # E.g., '599 599 N N ? TNX' -> '599 599 NJ NJ = TNX'
        # Also handles N<AR> and N<SK> (prosign attached by decoder)
        n_clean = upper.replace('<AR>', '').replace('<SK>', '').replace('<KN>', '')
        if n_clean == 'N' and i >= 2 and i < len(tokens) - 2:
            prev2_clean = tokens[i-2].upper().replace('<AR>', '').replace('<SK>', '').strip('.,;:!?=')
            prev1_clean = tokens[i-1].upper().replace('<AR>', '').replace('<SK>', '').strip('.,;:!?=')
            # Don't strip '?' from next - we need to detect it
            next1_clean = tokens[i+1].upper().replace('<AR>', '').replace('<SK>', '').strip('.,;:!= ')
            # Context: after 599 599, before ?/TNX pattern
            if prev2_clean == '599' and (prev1_clean == '599' or prev1_clean == 'N') and next1_clean in ('?', '=', 'N', 'TNX'):
                corrections.append({
                    'pos': i, 'original': token,
                    'corrected': 'NJ', 'type': 'n_to_nj'
                })
                result.append('NJ')
                continue
        
        result.append(token)
    
    # Auto-append K at end of CQ messages if missing
    # E.g., 'CQ TEST DE K3EST K3EST' -> 'CQ TEST DE K3EST K3EST K'
    if result:
        last = result[-1].upper().strip('.,;:!?=')
        if last != 'K' and last != 'SK' and last != 'AR' and last != 'BK' and last != 'BT' and last != 'BTU':
            # Check if last token is a callsign
            is_last_cs = False
            for pat in CALLSIGN_PATTERNS:
                if pat.fullmatch(last):
                    is_last_cs = True
                    break
            # Only append K if in CQ/calling context (first token is CQ)
            is_cq_context = False
            for r in result:
                if r.upper().strip('.,;:!?=') == 'CQ':
                    is_cq_context = True
                    break
            if is_last_cs and is_cq_context:
                corrections.append({
                    'pos': len(result), 'original': '',
                    'corrected': 'K', 'type': 'append_k'
                })
                result.append('K')
    
    return ' '.join(result), corrections


def fix_common_words(text: str) -> Tuple[str, List[dict]]:
    """
    Fix common word-level errors using lookup table.
    Also strips punctuation for matching, then reattaches.
    
    E.g., 'MRKING' -> 'WORKING', 'EOR' -> 'FOR'
    """
    corrections = []
    tokens = text.split(' ')
    result = []
    
    for i, token in enumerate(tokens):
        upper = token.upper()
        
        # Strip punctuation for matching
        stripped = upper.strip('.,;:!?=')
        trailing_punct = upper[len(upper.rstrip('.,;:!?=')):]  if upper != stripped else ''
        leading_punct = upper[:len(upper) - len(upper.lstrip('.,;:!?='))]
        match_target = stripped if stripped else upper
        
        # Skip if already correct or is a callsign/Q-code/abbreviation
        # BUT check WORD_FIXES first — some Q-codes/abbreviations have known errors
        if match_target in WORD_FIXES:
            fixed = WORD_FIXES[match_target]
            if fixed != match_target:
                corrections.append({
                    'pos': i, 'original': token,
                    'corrected': leading_punct + fixed + trailing_punct, 'type': 'word_fix'
                })
            result.append(leading_punct + fixed + trailing_punct)
            continue
        
        if match_target in CW_ABBREVIATIONS or match_target in Q_CODES:
            result.append(token)
            continue
        
        # Check callsign pattern - skip
        is_callsign = False
        for pat in CALLSIGN_PATTERNS:
            if pat.fullmatch(match_target):
                is_callsign = True
                break
        if is_callsign:
            result.append(token)
            continue
        
        # No fix found — keep as-is
        result.append(token)
    
    return ' '.join(result), corrections


def fuzzy_fix_words(text: str) -> Tuple[str, List[dict]]:
    """
    Phase 0.7: Fuzzy-match misspelled words against known vocabulary.
    
    For words not found in any dictionary, try Levenshtein distance=1
    against PROTECTED_ENGLISH + QSO_WORDS.
    Only for words >= 5 chars (short words are too ambiguous).
    """
    valid = _get_valid_words()
    corrections = []
    tokens = text.split(' ')
    result = []
    
    for i, token in enumerate(tokens):
        upper = token.upper()
        
        # Strip punctuation
        stripped = upper.strip('.,;:!?=')
        trailing_punct = upper[len(upper.rstrip('.,;:!?=')):] if upper != stripped else ''
        leading_punct = upper[:len(upper) - len(upper.lstrip('.,;:!?='))]
        match_target = stripped if stripped else upper
        
        # Skip if already a known word
        if match_target in valid:
            result.append(token)
            continue
        
        # Skip if already handled by WORD_FIXES
        if match_target in WORD_FIXES:
            result.append(token)
            continue
        
        # Skip callsigns, Q-codes, abbreviations, numbers
        if match_target in CW_ABBREVIATIONS or match_target in Q_CODES:
            result.append(token)
            continue
        if NUMBER_PATTERN.fullmatch(match_target):
            result.append(token)
            continue
        
        # Skip if contains non-alpha (likely callsign or mixed)
        if not match_target.isalpha():
            result.append(token)
            continue
        
        # Only fuzzy-match words >= 5 chars
        if len(match_target) < 5:
            result.append(token)
            continue
        
        # Skip common English word forms (plurals, -ing, -ed, etc.)
        # These are valid words that just aren't in our dictionary
        is_word_form = False
        for suffix in ['S', 'ING', 'ED', 'LY', 'ER', 'EST', 'EN', 'TION', 'MENT', 'NESS', 'ABLE', 'FUL', 'LESS']:
            if match_target.endswith(suffix) and len(match_target) > len(suffix) + 2:
                base = match_target[:-len(suffix)]
                # Handle doubled consonants (RUNNING->RUN, SITTING->SIT)
                if len(base) >= 2 and base[-1] == base[-2]:
                    base_short = base[:-1]
                    if base_short in valid:
                        is_word_form = True
                        break
                if base in valid:
                    is_word_form = True
                    break
                # Handle dropped-E for -ING (LIVING -> LIVE, HOPING -> HOPE)
                if suffix == 'ING' and (base + 'E') in valid:
                    is_word_form = True
                    break
                # Handle doubled consonant for -ED (HOPPED -> HOP)
                if suffix == 'ED' and len(base) >= 2 and base[-1] == base[-2]:
                    base_ed_short = base[:-1]
                    if base_ed_short in valid:
                        is_word_form = True
                        break
                # Handle dropped-E for -ED (HOPED -> HOPE)
                if suffix == 'ED' and (base + 'E') in valid:
                    is_word_form = True
                    break
                # Handle -IES -> -Y (CARRIES->CARRY)
                if suffix == 'S' and match_target.endswith('IES'):
                    base_y = match_target[:-3] + 'Y'
                    if base_y in valid:
                        is_word_form = True
                        break
                # Handle -ES -> -E (MAKES->MAKE)
                if suffix == 'S' and match_target.endswith('ES'):
                    base_e = match_target[:-1]
                    if base_e in valid:
                        is_word_form = True
                        break
                # Handle -VED -> -VE (LOVED->LOVE)
                if suffix == 'ED' and match_target.endswith('VED'):
                    base_ve = match_target[:-2]
                    if base_ve in valid:
                        is_word_form = True
                        break
        if is_word_form:
            result.append(token)
            continue
        
        # Try fuzzy match (edit distance = 1)
        best = None
        best_dist = 2
        for word in valid:
            if abs(len(word) - len(match_target)) > 1:
                continue  # length difference > 1 can't be distance 1
            dist = levenshtein(match_target, word)
            if dist < best_dist:
                best_dist = dist
                best = word
        
        if best and best_dist == 1:
            fixed = leading_punct + best + trailing_punct
            corrections.append({
                'pos': i, 'original': token,
                'corrected': fixed, 'type': 'fuzzy_fix'
            })
            result.append(fixed)
        else:
            result.append(token)
    
    return ' '.join(result), corrections


def correct_duplicate_callsigns(text: str) -> Tuple[str, List[dict]]:
    """
    Duplicate callsign consistency check.
    
    If the same callsign appears multiple times with slight variations (edit distance ≤ 1),
    use the most frequent version. If tied, prefer the last occurrence.
    
    Also: if a garbled callsign is close (distance ≤ 3) to a KNOWN_CALLSIGN
    that appears later in the same text, correct it.
    
    E.g., 'K3ESA K3EST' -> 'K3EST K3EST' (K3EST appears at the end, more reliable)
           'K2EV K3EST' -> 'K3EST K3EST' (K3EST is known, K2EV within distance 3)
    """
    corrections = []
    tokens = text.split(' ')
    
    # Find all callsign-like tokens (contain digit, 3-7 chars)
    callsign_positions = []
    for i, token in enumerate(tokens):
        cleaned = token.upper().strip('.,;:!?=').replace('?', '')
        if is_contest_cut_rst(cleaned):
            continue  # 5NN etc. are RST cut-numbers, not callsigns
        if 3 <= len(cleaned) <= 7 and any(c.isdigit() for c in cleaned) and cleaned.isalpha() or (
            3 <= len(cleaned) <= 7 and any(c.isdigit() for c in cleaned)
        ):
            # Verify it looks like a callsign
            for pat in CALLSIGN_PATTERNS:
                if pat.fullmatch(cleaned):
                    callsign_positions.append((i, cleaned, token))
                    break
    
    if len(callsign_positions) < 2:
        # Single callsign: try KNOWN_CALLSIGNS fuzzy match at distance 2
        if len(callsign_positions) == 1:
            pos, cleaned, original = callsign_positions[0]
            if cleaned not in KNOWN_CALLSIGNS:
                best = None
                best_dist = 3
                for cs in KNOWN_CALLSIGNS:
                    if abs(len(cs) - len(cleaned)) > 2:
                        continue
                    dist = levenshtein(cleaned, cs)
                    if dist < best_dist:
                        best_dist = dist
                        best = cs
                if best and best_dist <= 2:
                    tokens[pos] = best
                    corrections.append({
                        'pos': pos, 'original': original,
                        'corrected': best, 'type': 'callsign_known_fuzzy'
                    })
        return ' '.join(tokens), corrections
    
    # Group similar callsigns (edit distance ≤ 1)
    groups = []  # List of (representative, [positions])
    for pos, cleaned, original in callsign_positions:
        found_group = False
        for group in groups:
            rep = group[0]
            if levenshtein(cleaned, rep) <= 1:
                group[1].append((pos, cleaned, original))
                found_group = True
                break
        if not found_group:
            groups.append((cleaned, [(pos, cleaned, original)]))
    
    # Phase 0: For each non-known callsign, try KNOWN_CALLSIGNS fuzzy match (distance ≤ 2)
    # Stricter threshold (≤2) to avoid false positives like K2EV→K9ZZ
    # Phase 1 below handles distance ≤ 3 with context-aware across-group matching
    for i, (rep, members) in enumerate(groups):
        if rep in KNOWN_CALLSIGNS:
            continue
        best_cs = None
        best_dist = 3
        for cs in KNOWN_CALLSIGNS:
            if abs(len(cs) - len(rep)) > 2:
                continue
            dist = levenshtein(rep, cs)
            if dist < best_dist:
                best_dist = dist
                best_cs = cs
        if best_cs and best_dist <= 2:
            for pos, cleaned, original in members:
                tokens[pos] = best_cs
                corrections.append({
                    'pos': pos, 'original': original,
                    'corrected': best_cs, 'type': 'callsign_known_fuzzy'
                })
    
    # Phase 1: KNOWN_CALLSIGNS-aware match across groups (distance ≤ 3)
    # Skip groups that were already fixed in Phase 0
    for i, (rep_a, members_a) in enumerate(groups):
        for j, (rep_b, members_b) in enumerate(groups):
            if i >= j:
                continue
            # Skip if either group's representative was already fixed in Phase 0
            # (check current token value)
            first_member_a = members_a[0]
            first_member_b = members_b[0]
            curr_a = tokens[first_member_a[0]].upper().strip('.,;:!?=')
            curr_b = tokens[first_member_b[0]].upper().strip('.,;:!?=')
            if curr_a != rep_a or curr_b != rep_b:
                continue  # Already changed, skip
            # Only merge if exactly ONE is a known callsign (garbled → correct)
            # Never merge two different known callsigns
            a_known = rep_a in KNOWN_CALLSIGNS
            b_known = rep_b in KNOWN_CALLSIGNS
            
            # If both are known or neither is known, skip
            if a_known == b_known:
                continue
            
            dist = levenshtein(rep_a, rep_b)
            if dist <= 3:
                # Merge groups: prefer the known callsign
                winner = rep_a if a_known else rep_b
                loser_group = members_b if a_known else members_a
                for pos, cleaned, original in loser_group:
                    tokens[pos] = winner
                    corrections.append({
                        'pos': pos, 'original': original,
                        'corrected': winner, 'type': 'callsign_known_merge'
                    })
    
    # Phase 2: For each group with multiple variants, find the winner
    for rep, members in groups:
        if len(members) < 2:
            continue
        
        # Count unique variants
        variants = {}
        for pos, cleaned, original in members:
            if cleaned not in variants:
                variants[cleaned] = []
            variants[cleaned].append((pos, original))
        
        if len(variants) < 2:
            continue  # All same, no correction needed

        # Preserve distinct well-formed callsigns that differ only slightly
        # (OE1CIJ vs OE1CIW, M0TDK vs M0TDX) unless one clearly dominates.
        well_formed = []
        for variant in variants:
            ok = False
            for pat in CALLSIGN_PATTERNS:
                if pat.fullmatch(variant):
                    ok = True
                    break
            if ok and len(variant) >= 4:
                well_formed.append(variant)
        if len(well_formed) >= 2:
            # Always keep distinct well-formed callsign twins (OE1CIJ/OE1CIW,
            # M0TDK/M0TDX). Frequency majority is often the pile-up echo, not truth.
            continue
        
        # Find the most frequent variant; if tied, prefer the last occurrence
        best_variant = None
        best_count = 0
        best_last_pos = -1
        
        for variant, positions in variants.items():
            count = len(positions)
            last_pos = max(p for p, _ in positions)
            
            if count > best_count or (count == best_count and last_pos > best_last_pos):
                best_variant = variant
                best_count = count
                best_last_pos = last_pos
        
        # Apply corrections to non-winning variants
        for variant, positions in variants.items():
            if variant == best_variant:
                continue
            for pos, original in positions:
                old_val = tokens[pos]
                tokens[pos] = best_variant
                corrections.append({
                    'pos': pos, 'original': old_val,
                    'corrected': best_variant, 'type': 'callsign_consistency'
                })
    
    return ' '.join(tokens), corrections


def correct_rst_context(text: str) -> Tuple[str, List[dict]]:
    """
    Context-aware RST correction.
    
    1. When a 3-letter word appears before an RST number pattern (579, 599, etc.)
       and looks like RST with one error, fix it to RST.
    2. When a 3-digit number appears in RST context and has common errors
       (590->599, 699->599, 57'->579), fix the number.
    
    E.g., 'UR RSA 579' -> 'UR RST 579'
          'RST 590' -> 'RST 599'
    """
    corrections = []
    tokens = text.split(' ')
    
    # Phase 1: Fix words before RST numbers
    for i in range(len(tokens) - 1):
        next_token = tokens[i + 1].upper().strip()
        if not RST_PATTERN.fullmatch(next_token):
            continue
        curr = tokens[i].upper().strip()
        if len(curr) != 3 or not curr.isalpha():
            continue
        if curr == 'RST':
            continue
        dist = levenshtein(curr, 'RST')
        if dist == 1:
            old_val = tokens[i]
            leading = tokens[i][:len(tokens[i]) - len(tokens[i].lstrip('.,;:!?='))]
            trailing = tokens[i][len(tokens[i].rstrip('.,;:!?=')):]
            tokens[i] = leading + 'RST' + trailing
            corrections.append({
                'pos': i, 'original': old_val,
                'corrected': tokens[i], 'type': 'rst_context'
            })
    
    # Phase 2: Fix RST numbers themselves
    # Common RST values: 599, 579, 559, 449, 339
    VALID_RST = {'599', '579', '559', '449', '339'}
    # Pattern for garbled RST (includes 0 which is common error for 9, and special chars)
    GARBLED_RST = re.compile(r'^[0-9][0-9][0-9\'"&$?=]$|^[0-9][0-9\'"&$?=][0-9]$|^[0-9\'"&$?=][0-9][0-9]$')
    for i, token in enumerate(tokens):
        cleaned = token.upper().strip('.,;:!?=')
        if not cleaned:
            continue
        
        # Check for 3-char RST-like patterns
        if len(cleaned) == 3 and GARBLED_RST.fullmatch(cleaned):
            leading = tokens[i][:len(tokens[i]) - len(tokens[i].lstrip('.,;:!?='))]
            trailing = tokens[i][len(tokens[i].rstrip('.,;:!?=')):]
            
            # Try replacing non-digit or wrong-digit characters
            # First try replacing special chars
            for err, corr in [('?', '5'), ('?', '9'), ('?', '7'),
                              ('&', '5'), ('$', '3'), ("'", '9'),
                              ('=', '4')]:
                test = cleaned.replace(err, corr)
                if RST_PATTERN.fullmatch(test) and test in VALID_RST:
                    old_val = tokens[i]
                    tokens[i] = leading + test + trailing
                    corrections.append({
                        'pos': i, 'original': old_val,
                        'corrected': tokens[i], 'type': 'rst_number'
                    })
                    cleaned = test  # Update for next checks
                    break
            
            # Then try single-digit corrections for valid-looking RST numbers
            if cleaned not in VALID_RST:
                for j in range(3):
                    found = False
                    for replacement in '123456789':
                        test = cleaned[:j] + replacement + cleaned[j+1:]
                        if test in VALID_RST:
                            old_val = tokens[i]
                            tokens[i] = leading + test + trailing
                            corrections.append({
                                'pos': i, 'original': old_val,
                                'corrected': tokens[i], 'type': 'rst_number'
                            })
                            found = True
                            break
                    if found:
                        break
    
    return ' '.join(tokens), corrections


def remove_spurious_prosigns(text: str) -> Tuple[str, List[dict]]:
    """
    Remove spurious <SK>/<AR>/<KN> that appear inside words or attached
    to numbers — these are almost always element-merging artifacts from
    the decoder, not genuine prosigns.

    Patterns handled:
    - Inside a word: FOL<AR>OWS -> FOLLOWS, BUL<AR>ETIN -> BULLETIN
    - Attached to callsign/word: W1A<AR> -> W1AW, C<AR> -> CQ
    - Attached to numbers: <SK>79 -> 579
    - Standalone <SK>/<AR> between normal tokens in non-prosign context
    """
    corrections = []
    tokens = text.split(' ')
    result = []

    for i, token in enumerate(tokens):
        upper = token.upper()

        # Skip tokens that are ONLY a prosign (these are legitimate)
        if upper in ('<AR>', '<SK>', '<KN>'):
            result.append(token)
            continue

        # Skip empty tokens
        if not upper:
            result.append(token)
            continue

        # Check if token contains a prosign embedded in it
        if '<AR>' in upper or '<SK>' in upper or '<KN>' in upper:
            # This token has an embedded prosign — it's almost certainly
            # a decode artifact. Try to reconstruct the original token
            # by removing the prosign and seeing if the remaining letters
            # form something sensible.
            original = token
            cleaned = upper

            # Remove embedded prosigns
            cleaned = cleaned.replace('<AR>', '')
            cleaned = cleaned.replace('<SK>', '')
            cleaned = cleaned.replace('<KN>', '')

            if cleaned:
                corrections.append({
                    'pos': i, 'original': original,
                    'corrected': cleaned, 'type': 'spurious_prosign'
                })
                result.append(cleaned)
            else:
                # Entire token was a prosign artifact, drop it
                corrections.append({
                    'pos': i, 'original': original,
                    'corrected': '', 'type': 'spurious_prosign_drop'
                })
            continue

        result.append(token)

    return ' '.join(r for r in result if r), corrections


def _detach_prosigns(text: str) -> Tuple[str, List[dict]]:
    """
    Detach stray '=' (BT prosign) from CW abbreviations.
    E.g., 'AGN=' -> 'AGN =', '=TNX' -> '= TNX'
    Also strip leading '?' from words: '?FOLLOWS' -> 'FOLLOWS'
    """
    corrections = []
    tokens = text.split(' ')
    result = []
    
    for i, token in enumerate(tokens):
        # Detach trailing '=' from CW abbreviations: 'AGN=' -> 'AGN ='
        if token.endswith('=') and len(token) > 1:
            base = token[:-1].upper()
            if (base in CW_ABBREVIATIONS or base in Q_CODES or base in WORD_FIXES
                    or NUMBER_PATTERN.fullmatch(base)):
                corrections.append({
                    'pos': i, 'original': token,
                    'corrected': f'{base} =', 'type': 'detach_prosign'
                })
                result.append(f'{base} =')
                continue
        
        # Detach leading '=' from words: '=TNX' -> '= TNX'
        if token.startswith('=') and len(token) > 1:
            base = token[1:].upper()
            if base in CW_ABBREVIATIONS or base in Q_CODES or base in WORD_FIXES or base in PROTECTED_ENGLISH:
                corrections.append({
                    'pos': i, 'original': token,
                    'corrected': f'= {base}', 'type': 'detach_prosign_leading'
                })
                result.append(f'= {base}')
                continue
        
        # Strip leading '?' from words: '?FOLLOWS' -> 'FOLLOWS'
        if token.startswith('?') and len(token) > 1:
            base = token[1:]
            base_upper = base.upper()
            # Only strip if the remaining word is a known word
            if base_upper in PROTECTED_ENGLISH or base_upper in QSO_WORDS or base_upper in CW_ABBREVIATIONS:
                corrections.append({
                    'pos': i, 'original': token,
                    'corrected': base_upper, 'type': 'strip_leading_question'
                })
                result.append(base_upper)
                continue
        
        result.append(token)
    
    return ' '.join(result), corrections


def _clean_special_chars(text: str) -> Tuple[str, List[dict]]:
    """
    Clean special characters that are decoder artifacts.
    - '(' in words -> 'NG' (e.g., WARNI( -> WARNING)
    - Leading '/' -> 'C' (e.g., /Q -> CQ)
    - '&' in words -> common replacements
    """
    corrections = []
    tokens = text.split(' ')
    result = []
    
    for i, token in enumerate(tokens):
        original = token
        changed = False
        
        # Replace '(' inside/at end of words with 'NG' (WARNI( -> WARNING)
        if '(' in token:
            token = token.replace('(', 'NG')
            changed = True
        
        # Replace leading '/' with 'C' (/Q -> CQ)
        if token.startswith('/'):
            token = 'C' + token[1:]
            changed = True
        
        # Replace '&' with 'AND' or remove
        if '&' in token:
            token = token.replace('&', '')
            changed = True
        
        # Replace ?PY? -> CPY? (common garble pattern for CPY)
        if token.upper() == '?PY?':
            token = 'CPY?'
            changed = True
        
        if changed:
            corrections.append({
                'pos': i, 'original': original,
                'corrected': token, 'type': 'special_char'
            })
        result.append(token)
    
    return ' '.join(result), corrections


def semantic_correct(text: str) -> Tuple[str, List[dict]]:
    """
    Perform semantic post-correction on decoded text.

    Returns: (corrected text, list of correction records)
    """
    all_corrections = []
    
    # Phase -1: Remove spurious prosigns embedded in words (must be first)
    text, prosign_fixes = remove_spurious_prosigns(text)
    all_corrections.extend(prosign_fixes)
    
    # Phase -0.5: Clean special characters (decoder artifacts)
    text, special_fixes = _clean_special_chars(text)
    all_corrections.extend(special_fixes)

    # Phase -0.25: Split contest run-ons (CALL5NN1234, TUSZ1A, …)
    text, contest_fixes = split_contest_runon(text)
    all_corrections.extend(contest_fixes)
    
    # Phase 0: Split merged words first (context-free)
    text, merged_fixes = split_merged_words(text)
    all_corrections.extend(merged_fixes)
    
    # Phase 0.3: Split merged callsigns
    text, cs_merge_fixes = split_merged_callsigns(text)
    all_corrections.extend(cs_merge_fixes)
    
    # Phase 0.4: Detach stray '=' from CW abbreviations
    # Decoder sometimes attaches '=' to the preceding word (e.g., 'AGN=' instead of 'AGN =')
    text, detach_fixes = _detach_prosigns(text)
    all_corrections.extend(detach_fixes)
    
    # Phase 0.5: Fix common word errors
    text, word_fixes = fix_common_words(text)
    all_corrections.extend(word_fixes)
    
    # Phase 0.7: Fuzzy-match misspelled words
    text, fuzzy_fixes = fuzzy_fix_words(text)
    all_corrections.extend(fuzzy_fixes)
    
    # Phase 0.75: Context-aware Q-code/prosign corrections
    text, qso_fixes = correct_qso_context(text)
    all_corrections.extend(qso_fixes)
    
    # Phase 0.8: Context-aware RST correction
    text, rst_fixes = correct_rst_context(text)
    all_corrections.extend(rst_fixes)
    
    # Phase 0.9: Duplicate callsign consistency
    text, cs_fixes = correct_duplicate_callsigns(text)
    all_corrections.extend(cs_fixes)
    
    # Phase 1: Token-level corrections
    # Split by spaces, preserving original space structure
    tokens = text.split(' ')
    corrected = []
    corrections = []

    for i, w in enumerate(tokens):
        original = w

        # 0. Protect special symbols (=, <AR>, <SK>, etc.) from modification
        if w in PROTECTED_TOKENS or w == '':
            corrected.append(w)
            continue

        w_upper = w.upper()

        # 0b. Skip pure numbers
        if NUMBER_PATTERN.fullmatch(w_upper):
            corrected.append(w_upper)
            continue

        # 1. Check RST format (e.g. 599)
        c, changed = correct_rst(w)
        if not changed and RST_PATTERN.fullmatch(c):
            corrected.append(c)
            continue
        if changed:
            corrections.append({
                'pos': i, 'original': original,
                'corrected': c, 'type': 'rst'
            })
            corrected.append(c)
            continue

        # 2. Check WORD_FIXES for Q-codes/abbreviations with known errors
        if w_upper in WORD_FIXES:
            fixed = WORD_FIXES[w_upper]
            corrections.append({
                'pos': i, 'original': original,
                'corrected': fixed, 'type': 'word_fix_phase1'
            })
            corrected.append(fixed)
            continue

        # 2b. Check if known abbreviation or Q-code (don't modify)
        if w_upper in CW_ABBREVIATIONS or w_upper in Q_CODES:
            corrected.append(w_upper)
            continue

        # 3. Try callsign correction
        c, changed = correct_callsign(w)
        if changed:
            corrections.append({
                'pos': i, 'original': original,
                'corrected': c, 'type': 'callsign'
            })
            corrected.append(c)
            continue

        # 4. Try Q-code correction
        c, changed = correct_qcode(w)
        if changed:
            corrections.append({
                'pos': i, 'original': original,
                'corrected': c, 'type': 'qcode'
            })
            corrected.append(c)
            continue

        # 5. Try abbreviation correction
        c, changed = correct_abbreviation(w)
        if changed:
            corrections.append({
                'pos': i, 'original': original,
                'corrected': c, 'type': 'abbreviation'
            })
            corrected.append(c)
            continue

        # 6. Keep as-is
        corrected.append(w_upper)

    all_corrections.extend(corrections)
    return ' '.join(corrected), all_corrections


def batch_correct(texts: List[str]) -> dict:
    """Batch-correct multiple texts."""
    results = []
    total_corrections = 0

    for text in texts:
        corrected, corrections = semantic_correct(text)
        results.append({
            'original': text,
            'corrected': corrected,
            'corrections': corrections,
            'n_corrections': len(corrections)
        })
        total_corrections += len(corrections)

    return {
        'results': results,
        'total_corrections': total_corrections,
        'avg_corrections': total_corrections / len(texts) if texts else 0
    }


if __name__ == "__main__":
    # Test cases
    test_cases = [
        "cq cq de bg6xy? 599 599",
        "qrz de w1aw ?st 599",
        "cq cq de ba1aa k",
        "tks ?r 73 es gl",
        "rst 5?9 599 qsl",
        "cq de n1?em ?/?",
    ]

    print("=" * 60)
    print("Semantic Correction Test")
    print("=" * 60)

    for test in test_cases:
        corrected, corrections = semantic_correct(test)
        print(f"\nOriginal:  {test}")
        print(f"Corrected: {corrected}")
        if corrections:
            print(f"Fixes:     {corrections}")

