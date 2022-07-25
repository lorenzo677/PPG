#%%
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from scipy.interpolate import interp1d
import numpy as np
# %%
def extract_from_df(dataframe, list_of_feature):
    df = pd.DataFrame()
    for i, feature in enumerate(list_of_feature):
        df.insert(i , feature, dataframe[feature])
    return df

def save_pickle(filename, dataframe):
    with open(filename, 'wb') as file:
        pickle.dump(dataframe, file)


def m_avg(t, x, w): return (np.asarray([t[i] for i in range(w, len(x) - w)]),
                            np.convolve(x, np.ones((2*w + 1, )) / (2*w + 1),
                                        mode='valid'))


def detect_peaks(signal, mov_avg):
    window = []
    peaklist = []
    for (i, datapoint), roll in zip(enumerate(signal), mov_avg):
        if (datapoint <= roll) and (len(window) <= 1):
            continue
        elif (datapoint > roll):
            window.append(datapoint)
        else:
            beatposition = i - len(window) + np.argmax(window)
            peaklist.append(beatposition)
            window = []

    return peaklist, [signal[x] for x in peaklist]


# def db_to_dataframe(filename):
#     """
#     Load the json file obtained through create_db into a pandas dataframe

#     Parameters
#     ----
#     filename: string; the full path to the json file, complete with extension

#     Return
#     ----
#     d: DataFrame; the dataframe containing the elements stored in the file
#     """
#     d = json.load(open(filename))
#     d = pd.DataFrame(d).T
#     d = d.set_index(np.arange(len(d)))
#     return d


def find_x_of_minima(time, signal):
    """
    find index position of local minima whose amplitude is under a certain
    moving threshold

    Parameters
    ----
    time: numerical 1-D array-like; basically the x axis of the curve whose
    minima will be found

    signal: numerical 1-D array-like; basically the y axis of the curve whose
    minima will be found

    Return
    ----
    final_peaks: list; the list containing the index positions of signal minima
    """

    # -1* is used to find minima instead of maxima
    sign = -1*np.asarray(signal)

    # using time to extrapolate sampling rate
    srate = len(time)/(max(time)-min(time))
    peaks = np.arange(len(sign))  # initializing peaks index

    # different widths used for moving window, the unit is referred to 1 srate
    for i in np.array([.5, 1., 1.5, 2., 3.]):
        mt, mov_avg = m_avg(time, sign, int(srate*i))

        # use len_filler to make mov_avg the same size as sign
        len_filler = np.zeros((len(sign)-len(mov_avg))//2) + np.mean(sign)
        mov_avg = np.insert(mov_avg, 0, len_filler)
        mov_avg = np.append(mov_avg, len_filler)

        peaklist, sign_peak = detect_peaks(sign, mov_avg)

        # keeping only peaks detected with all 5 different windows
        peaks = np.intersect1d(peaks, peaklist)

    # first element can't be a correct local extrema, it has no points before
    if(peaks[0] == 0):
        peaks = np.delete(peaks, 0)

    # last element can't be a correct local extrema, it has no points after
    if(peaks[-1] == len(sign)-1):
        peaks = np.delete(peaks, -1)

    # peak checking: rejecting lower peaks where RR distance is too small
    final_peaks = []  # definitive peak positions container
    last_peak = -1  # parameter to avoid undesired peaks still in final_peaks
    for p in peaks:
        if p <= last_peak:
            continue

        evaluated_peaks = [g for g in peaks if p <= g <= srate*.5+p]
        last_peak = max(evaluated_peaks)
        final_peaks.append(evaluated_peaks[np.argmin(
            [sign[x] for x in evaluated_peaks])])

    final_peaks = np.unique(final_peaks)  # to avoid repetitions

    return final_peaks
# %%
filepath_input = 'db_after_notch.json'
list_of_feature = ('signal', 'time', 'age', 'quality', 'bpm')
filepath_output = '/Users/lorenzo/Desktop/PPG/db_ppg_peaks.pickle'
# %%
raw_df = pd.read_json(filepath_input, orient='index')

df = extract_from_df(raw_df, list_of_feature)
# %%
signal_patient = []
quality_threshold = 0.005
# idx_not_valid_signal = 3069

#%%
df = df[np.logical_and(df['quality']>=0, df['quality']<quality_threshold)]
df = df[df['age'] != 0]
df = df.reset_index(drop=True)
#df = df.drop(columns=['quality'])

N_PEOPLE = len(df['signal'])


#%%

final_peaks = []
final_labels = []
separate_guys = []

cont = 0
for guy in range(N_PEOPLE):  # test_idx # range(len(d)):
    #print(cont/len(d))       # same
    cont += 1
    separate_guys.append(len(final_peaks))
    sample = df.iloc[guy]  # d.loc[guy] # d.iloc[guy] #con loc non funziona
    splits = find_x_of_minima(sample.time, sample.signal)
    peaks = np.split(sample.signal, splits[5::15])[1:-1]
    times = np.split(sample.time, splits[5::15])[1:-1]
    labels = sample.drop(['signal', 'time'])
    try:
        new_peaks = list(map(lambda tx, sy: interp1d(tx, sy, kind="cubic")(
            np.linspace(min(tx), max(tx), 1024)), times, peaks))
        final_peaks = final_peaks + new_peaks
        final_labels = final_labels + [labels]*len(new_peaks)
    except ValueError:
        pass
print("ended")

#%%
new_df=pd.DataFrame(final_labels)
new_df['signal'] =final_peaks
new_df = new_df.reset_index(drop=True)

print(new_df)
# %%
save_pickle(filepath_output, new_df)
# %%
