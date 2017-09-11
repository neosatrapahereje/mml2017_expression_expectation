import numpy as np
from decimal import Decimal
from scipy.interpolate import interp1d
from scipy.misc import derivative
# import matplotlib.pyplot as plt

ONSETWISE_TARGETS = ['velocity_trend', 'ibi', 'dvelocity_trend', 'dibi']
NOTEWISE_TARGETS = ['velocity_dev', 'timing', 'articulation']
M_INPUTS = ['m_pitch', 'm_probability', 'm_ic', 'm_entropy']
C_INPUTS = ['c_pitch', 'c_vi_1', 'c_vi_2', 'c_vi_3', 'c_probability',
            'c_ic', 'c_entropy']
INPUTS = M_INPUTS + C_INPUTS + ['h_pitch', 'beat_phase', 'down_beat',
                                'sec_strong_beat', 'weak_beat']

SCORE_FEATURES = ['h_pitch', 'c_pitch', 'm_pitch',
                  'c_vi_1', 'c_vi_2', 'c_vi_3',
                  'beat_phase', 'down_beat', 'sec_strong_beat', 'weak_beat']

EXPECTANCY_FEATURES = ['m_probability', 'm_ic', 'm_entropy',
                       'm_probability', 'm_ic', 'm_entropy']

PRETTY_INPUT_NAMES = dict(
    m_pitch='$pitch_m$',
    m_probability='$p_m$',
    m_ic='$IC_m$',
    m_entropy='$H_m$',
    c_pitch='$pitch_l$',
    c_vi_1='$vi_1$',
    c_vi_2='$vi_2$',
    c_vi_3='$vi_3$',
    c_probability='$p_c$',
    c_ic='$IC_c$',
    c_entropy='$H_c$',
    h_pitch='$pitch_h$',
    beath_phase='$b_\phi$',
    down_beat='$b_d$',
    sec_strong_beat='$b_s$',
    weak_beat='$b_w$')


def str2bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False


def _dec_round(x, decimals=0):

    if decimals == 0:
        return np.round(x, decimals=0)
    else:
        exp = ''.join(['0.'] + (decimals - 1) * ['0'] + ['1'])
        rounded = Decimal(str(x)).quantize(Decimal(exp))
        return float(rounded)


def dec_round(x, decimals=0):

    if isinstance(x, float):
        return _dec_round(x, decimals)

    else:
        shape = x.shape

        rounded = np.array(
            [_dec_round(v, decimals) for v in x.reshape(-1)])

        return rounded.reshape(shape)


def get_onset_idxs(onsets, score_onsets, tol=0.1):
    a = []

    for o in onsets[onsets.argsort()]:
        diff = abs(score_onsets - o)
        wix = int(diff.argmin())
        a.append(wix)
    return np.array(a).flatten()


def parse_score(fn, remove_grace_notes=True):
    with open(fn) as f:
        data = f.read()

    score_position = []
    midi_pitch = []
    score_duration = []
    performed_onset = []
    performed_duration = []
    midi_velocity = []
    is_soprano = []

    for line in data.splitlines():

        # * After the score position
        # * midi pitch
        # * score duration in beats
        # * performed onset in seconds
        # * performed duration in seconds
        # * midi velocity
        # * flag indicating whether note is a soprano note

        a = line.split(' ')

        n_notes = int(len(a[1:]) / 6.)

        s_duration = np.array(
            [float(d) for d in a[n_notes + 1: 2 * (n_notes) + 1]])

        if remove_grace_notes:
            grace_note_idx = s_duration == 0
            valid_idx = ~grace_note_idx
            s_duration = s_duration[valid_idx]
        else:
            valid_idx = np.ones(len(s_duration)).astype(bool)

        if len(s_duration) > 0:
            score_duration.append(s_duration)

            score_position.append(float(a[0]))

            pitch = np.array([float(p) for p in a[1:n_notes + 1]])
            pitch = pitch[valid_idx]
            midi_pitch.append(pitch)

            onset = np.array(
                [float(d) for d in a[2 * n_notes + 1: 3 * n_notes + 1]])
            onset = onset[valid_idx]
            performed_onset.append(onset)

            p_duration = np.array(
                [float(d) for d in a[3 * n_notes + 1: 4 * n_notes + 1]])
            p_duration = p_duration[valid_idx]
            performed_duration.append(p_duration)

            velocity = np.array(
                [float(d) for d in a[4 * n_notes + 1: 5 * n_notes + 1]])
            velocity = velocity[valid_idx]
            midi_velocity.append(velocity)

            flag = np.array(
                [str2bool(d) for d in a[5 * n_notes + 1: 6 * n_notes + 1]])
            flag = flag[valid_idx]
            is_soprano.append(flag)

    return dict(
        score_position=np.array(score_position),
        pitch=np.array(midi_pitch),
        score_duration=np.array(score_duration),
        performed_onset=np.array(performed_onset),
        performed_duration=np.array(performed_duration),
        midi_velocity=np.array(midi_velocity),
        is_soprano=np.array(is_soprano))


def get_targets(score):
    score_onsets = score['score_position']
    velocity_trend = np.array([np.max(v)
                               for v in score['midi_velocity'] / 127.])

    velocity_fun = interp1d(score_onsets,
                            velocity_trend,
                            kind='linear',
                            bounds_error=False,
                            fill_value=(velocity_trend[0], velocity_trend[-1]))

    dvel = derivative(velocity_fun, score_onsets, dx=0.5)

    velocity_dev = np.array(
        [pv - mv for pv, mv in zip(score['midi_velocity'] / 127., velocity_trend)])

    onset_trend = np.array([np.mean(so) for so in score['performed_onset']])

    last_offset = np.max(score['performed_duration'][-1]) + onset_trend[-1]

    last_s_offset = np.max(score['score_duration'][-1]) + score_onsets[-1]

    onset_fun = interp1d(np.append(score_onsets, last_s_offset),
                         np.append(onset_trend, last_offset),
                         kind='linear',
                         # bounds_error=False,
                         fill_value='extrapolate')
    ibi = derivative(onset_fun, score_onsets, dx=0.5)
    mean_ibi = ibi.mean()
    ibi /= mean_ibi
    dibi = derivative(onset_fun, score_onsets, n=2, dx=0.5) / mean_ibi

    onset_trend_rec = np.cumsum(
        np.r_[onset_trend[0],
              ibi[:-1] * mean_ibi * np.diff(score_onsets)])

    onset_dev = (np.array(
        [(po - mo) / mean_ibi
         for po, mo in zip(score['performed_onset'], onset_trend_rec)]))

    articulation = (score['performed_duration'] /
                    (score['score_duration'] * mean_ibi + 1e-7))

    return dict(
        velocity_dev=velocity_dev,
        velocity_trend=velocity_trend,
        dvelocity_trend=dvel,
        ibi=ibi,
        dibi=dibi,
        timing=onset_dev,
        articulation=articulation,
        mean_ibi=mean_ibi)


def parse_chord_file(fn):
    # * After the score position, there is:
    # * score duration of chord in beats
    # * vertical interval class combination (containing between one and three
    #   interval classes)
    # * performed onset in seconds
    # * performed duration in seconds

    with open(fn) as f:
        chord_raw_data = f.read()

    score_position = []
    score_duration = []
    vertical_interval_class = []
    performed_onset = []
    performed_duration = []
    for l in chord_raw_data.splitlines():
        ch_l = l.split(' ')
        ch_l.remove('')

        score_position.append(float(ch_l[0]))
        score_duration.append(float(ch_l[1]))
        vertical_interval_class.append(
            [float(vi) for vi in ch_l[2:-2]])
        performed_onset.append(float(ch_l[-2]))
        performed_duration.append(float(ch_l[-1]))

    return dict(
        score_position=np.array(score_position),
        score_duration=np.array(score_duration),
        vertical_interval_class=np.array(vertical_interval_class),
        performed_onset=np.array(performed_onset),
        performed_duration=np.array(performed_duration)
    )


def parse_melody_file(fn):

    mel_data = np.loadtxt(fn)

    score_perf_attribute_names = [
        'score_position', 'score_duration', 'channel',
        'pitch', 'velocity', 'performed_onsets', 'performed_duration']

    return dict(zip(score_perf_attribute_names, mel_data.T))


def load_piece(piece, idyom_model, interpolate=True, timesigs=None,
               c_idyom_model='onsets_vintcc_ltm'):
    print piece
    melody_score = parse_melody_file(
        'data/txt_melodies/{0}.txt'.format(piece))
    chord_score = parse_chord_file('data/txt_chords/{0}.txt'.format(piece))
    score = parse_score('data/txt_from_match_files/{0}.txt'.format(piece))

    if timesigs is None:
        timesigs = np.loadtxt('data/mozart_timesigs.txt', dtype=str)

    metrical_features = compute_metrical_features(timesigs, piece,
                                                  score['score_position'])

    c_idyom_features = np.loadtxt(
        'data/idyom_models/{0}/{1}.txt'.format(c_idyom_model, piece))
    m_idyom_features = np.loadtxt(
        'data/idyom_models/{0}/{1}.txt'.format(idyom_model, piece))

    onset_c_data = len(c_idyom_features) == len(score['score_position'])

    targets = get_targets(score)

    m_idx = get_onset_idxs(
        melody_score['score_position'], score['score_position'])
    c_idx = get_onset_idxs(
        chord_score['score_position'], score['score_position'])

    if onset_c_data:

        c_pitch = np.zeros((len(score['score_position']), 4))

        # c_pitch[:, 0] = np.array(
        #     [np.min(p) / 127. for p in score['pitch']])

        for ci, p, vi in zip(c_idx, score['pitch'][c_idx],
                             chord_score['vertical_interval_class']):
            c_pitch[np.min(ci), 0] = np.min(p) / 127.
            c_pitch[np.min(ci), 1:len(vi) + 1] = np.array(vi) / 11.

    else:

        c_pitch = np.zeros((len(chord_score['score_position']), 4))

        for i, (p, vi) in enumerate(zip(score['pitch'][c_idx],
                                        chord_score['vertical_interval_class'])):

            c_pitch[i, 0] = np.min(p) / 127.
            c_pitch[i, 1:len(vi) + 1] = np.array(vi) / 11.

    c_data = np.hstack((c_pitch, c_idyom_features))

    c_targets = np.column_stack([targets[tn][c_idx]
                                 for tn in ONSETWISE_TARGETS])

    # Use pitch from txt_melody
    m_data = np.hstack(
        (melody_score['pitch'].reshape(-1, 1) / 127., m_idyom_features))

    m_targets = np.column_stack([targets[tn][m_idx]
                                 for tn in ONSETWISE_TARGETS])

    combined_data = np.zeros((len(score['score_position']),
                              c_data.shape[1] + m_data.shape[1] + 1))

    if interpolate:
        print 'Interpolating expectation features...'
        m_prob_fun = interp1d(melody_score['score_position'],
                              m_idyom_features[:, 0],
                              kind='zero',
                              bounds_error=False,
                              fill_value=(0, m_idyom_features[-1, 0]))
        m_ic_fun = interp1d(melody_score['score_position'],
                            m_idyom_features[:, 1],
                            kind='zero',
                            bounds_error=False,
                            fill_value=(0, m_idyom_features[-1, 1]))
        m_entropy_fun = interp1d(melody_score['score_position'],
                                 m_idyom_features[:, 2],
                                 kind='zero',
                                 bounds_error=False,
                                 fill_value=(0, m_idyom_features[-1, 2]))
        if onset_c_data:
            csp = score['score_position']
        else:
            csp = chord_score['score_position']

        c_prob_fun = interp1d(csp,
                              c_idyom_features[:, 0],
                              kind='zero',
                              bounds_error=False,
                              fill_value=(0, c_idyom_features[-1, 0]))
        c_ic_fun = interp1d(csp,
                            c_idyom_features[:, 1],
                            bounds_error=False,
                            kind='zero',
                            fill_value=(0, c_idyom_features[-1, 1]))
        c_entropy_fun = interp1d(csp,
                                 c_idyom_features[:, 2],
                                 bounds_error=False,
                                 kind='zero',
                                 fill_value=(0, c_idyom_features[-1, 2]))

        m_prob = m_prob_fun(score['score_position'])
        m_ic = m_ic_fun(score['score_position'])
        m_entropy = m_entropy_fun(score['score_position'])

        c_prob = c_prob_fun(score['score_position'])
        c_ic = c_ic_fun(score['score_position'])
        c_entropy = c_entropy_fun(score['score_position'])

        combined_data[:, 1] = m_prob
        combined_data[:, 2] = m_ic
        combined_data[:, 3] = m_entropy

        combined_data[:, 8] = c_prob
        combined_data[:, 9] = c_ic
        combined_data[:, 10] = c_entropy

    if onset_c_data:
        combined_data[:, 4:-1] = c_data

    for i, (so, p) in enumerate(zip(score['score_position'], score['pitch'])):

        if i in m_idx:
            # TODO: there is a weird thing going on...
            mi = np.where(m_idx == i)[0]
            combined_data[i, :4] = m_data[np.min(mi)]

        if i in c_idx and not onset_c_data:
            ci = np.where(c_idx == i)[0]
            combined_data[i, 4:-1] = c_data[np.min(ci)]
        combined_data[i, -1] = np.max(p) / 127.

    combined_data = np.hstack((combined_data, metrical_features))

    combined_targets = np.column_stack([targets[tn]
                                        for tn in ONSETWISE_TARGETS])

    return dict(
        m_data=m_data,
        m_targets=m_targets,
        c_data=c_data,
        c_targets=c_targets,
        combined_data=combined_data,
        combined_targets=combined_targets)


def test1():
    import matplotlib.pyplot as plt
    pieces = ['kv283_2']
    idyom_model = 'cpitch_selection_ltm'
    inputs = None
    targets = [0, 1]
    data_type = 'combined'

    if data_type == 'combined':
        features = INPUTS
    elif data_type == 'm':
        features = M_INPUTS
    elif data_type == 'c':
        features = C_INPUTS

    input_names = features
    if inputs:
        input_names = [input_names[i] for i in inputs]

    output_names = [ONSETWISE_TARGETS[i] for i in targets]

    feature_idx = np.array([features.index(i) for i in input_names])
    target_idx = np.array([ONSETWISE_TARGETS.index(i) for i in output_names])

    X = []
    Y = []

    for piece in pieces:
        piece_dict = load_piece(piece, idyom_model, interpolate=True)

        x = piece_dict['{0}_data'.format(data_type)][:, feature_idx]
        y = piece_dict['{0}_targets'.format(data_type)][:, target_idx]

        y = (y - y.mean(0, keepdims=True)) / (y.std(0, keepdims=True))

        X.append(x)
        Y.append(y)

    X = np.array(X)
    Y = np.array(Y)
    pieces = np.array(pieces)

    plt.imshow(x[:100].T, cmap='gray', aspect='auto', interpolation='nearest',
               origin='lower')
    plt.show()


def compute_metrical_features(timesigs, piece, score_onsets):

    # get index of the piece
    p_idx = int(np.where(timesigs[:, 0] == piece)[0])

    # time signature
    timesig = timesigs[p_idx][1]

    # get number of beats
    n_beats = float(timesig.split('/')[0])

    beat_phase = np.mod(score_onsets, n_beats)

    beat_strength = np.zeros((len(score_onsets), 3))

    # downbeat, secondary strong, weak beat
    strong_beat_idxs = np.where(beat_phase == 0)[0]
    beat_strength[strong_beat_idxs, 0] = 1.

    if timesig == '4/4':

        sec_strong_beat_idxs = np.where(beat_phase == 2)[0]
        beat_strength[sec_strong_beat_idxs, 1] = 1.
        weak_beat_idxs = np.where(~np.logical_or(beat_phase == 0,
                                                 beat_phase == 2))[0]
        beat_strength[weak_beat_idxs, 2] = 1.

    elif timesig == '2/4':

        weak_beat_idxs = np.where(beat_phase != 0)[0]
        beat_strength[weak_beat_idxs, 2] = 1.

    elif timesig == '3/4':

        weak_beat_idxs = np.where(beat_phase != 0)[0]
        beat_strength[weak_beat_idxs, 2] = 1.

    elif timesig == '6/8':

        sec_strong_beat_idxs = np.where(beat_phase == 3)[0]
        beat_strength[sec_strong_beat_idxs, 1] = 1.
        weak_beat_idxs = np.where(~np.logical_or(beat_phase == 0,
                                                 beat_phase == 3))[0]
        beat_strength[weak_beat_idxs, 2] = 1.

    # compute beat phase as Xia et al.
    beat_phase /= n_beats

    return np.column_stack((beat_phase, beat_strength))

if __name__ == '__main__':

    fn = 'data/mozart_timesigs.txt'
    score_onsets = np.arange(10)

    timesigs = np.loadtxt(fn, dtype=str)

    piece = 'kv333_2'
    idyom_model = 'cpitch_contour_ltm'

    interp = load_piece(piece, idyom_model, True, timesigs)
    non_interp = load_piece(piece, idyom_model, False, timesigs)

    non_idx = np.where(non_interp['combined_data'][
                       :, 1] != interp['combined_data'][:, 1])[0]
    zero_idx = np.where(non_interp['combined_data'][:, 1] == 0)[0]
