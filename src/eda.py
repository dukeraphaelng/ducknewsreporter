import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS

from scipy import spatial
from scipy.spatial.distance import jensenshannon
from sklearn.feature_selection import f_classif

from non_latent_features import NonLatentFeatures
from textual_relevance import TextualRelevance
from preprocess import Preprocessor

def extract_non_latent(row):
    total_dict = {}
    for k in ['div_NOUN_sum', 'div_NOUN_percent', 'div_VERB_sum', 'div_VERB_percent', 'div_ADJ_sum', 'div_ADJ_percent', 'div_ADV_sum', 'div_ADV_percent', 'div_LEX_sum', 'div_LEX_percent', 'div_CONT_sum', 'div_CONT_percent', 'div_FUNC_sum', 'div_FUNC_percent', 'pron_FPS_sum', 'pron_FPS_percent', 'pron_FPP_sum', 'pron_FPP_percent', 'pron_STP_sum', 'pron_STP_percent', 'quant_NOUN_sum', 'quant_NOUN_percent', 'quant_VERB_sum', 'quant_VERB_percent', 'quant_ADJ_sum', 'quant_ADJ_percent', 'quant_ADV_sum', 'quant_ADV_percent', 'quant_PRON_sum', 'quant_PRON_percent', 'quant_DET_sum', 'quant_DET_percent', 'quant_NUM_sum', 'quant_NUM_percent', 'quant_PUNCT_sum', 'quant_PUNCT_percent', 'quant_SYM_sum', 'quant_SYM_percent', 'quant_PRP_sum', 'quant_PRP_percent', 'quant_PRP$_sum', 'quant_PRP$_percent', 'quant_WDT_sum', 'quant_WDT_percent', 'quant_CD_sum', 'quant_CD_percent', 'quant_VBD_sum', 'quant_VBD_percent', 'quant_STOP_sum', 'quant_STOP_percent', 'quant_LOW_sum', 'quant_LOW_percent', 'quant_UP_sum', 'quant_UP_percent', 'quant_NEG_sum', 'quant_NEG_percent', 'quant_QUOTE_sum', 'quant_NP_sum', 'quant_CHAR_sum', 'quant_WORD_sum', 'quant_SENT_sum', 'quant_SYLL_sum', 'senti_!_sum', 'senti_!_percent', 'senti_?_sum', 'senti_?_percent', 'senti_CAPS_sum', 'senti_CAPS_percent', 'senti_POL_sum', 'senti_SUBJ_sum', 'avg_chars_per_word_sum', 'avg_words_per_sent_sum', 'avg_claus_per_sent_sum', 'avg_puncts_per_sent_sum', 'med_st_ALL_sum', 'med_st_NP_sum', 'read_gunning-fog_sum', 'read_coleman-liau_sum', 'read_dale-chall_sum', 'read_flesch-kincaid_sum', 'read_linsear-write_sum', 'read_spache_sum', 'read_automatic_sum', 'read_flesch_sum']:
        for content_k in ("content", "ctx1_content", "ctx2_content", "ctx3_content"):
            total_dict[content_k + '_' + k] = 0
        
    for key in ("content", "ctx1_content", "ctx2_content", "ctx3_content"):
        if row[key] is None:
            continue
        if not isinstance(row[key], str) and pd.isnull(row[key]):
            continue

        non_latent_dict = NonLatentFeatures(row[key]).output_all()        
        for k, v in non_latent_dict.items():
            total_dict[key + '_' + k] = v
    return pd.Series(total_dict.values(), total_dict.keys())

def non_latent_cosine_dist_func(columns):
    def non_latent_cosine_dist(row):
        predict_vec = np.array([row[col] for col in columns])
        context_vecs = []
        # get all vec of context vectors here
        for type_ in ["ctx1_content", "ctx2_content", "ctx3_content"]:
            if isinstance(row[type_], str) and len(row[type_]) > 0:
                context_vec = np.array([row[type_ + col[7:]] for col in columns])
                context_vecs.append(context_vec)

        val = np.mean([1 - spatial.distance.cosine(predict_vec, context_vec) for context_vec in context_vecs])

        return pd.Series([val], ['non_latent_cosine_dist'])
    
    return non_latent_cosine_dist

def apply_textual_relevance(df):
    pp = Preprocessor()

    for col in ["content", "ctx1_content", "ctx2_content", "ctx3_content"]:
        df[col + '_token'] = df[col].apply(pp.tokenize_opt)

    tfidf_1_1 = TextualRelevance('tfidf', df.content, ngram_range=(1, 1))
    tfidf_1_2 = TextualRelevance('tfidf', df.content, ngram_range=(1, 2))
    word2vec = TextualRelevance('word2vec')

    tf_idf_1_1_cosine_dist = []
    tf_idf_1_1_word_app = []
    tf_idf_1_1_matching = []

    tf_idf_1_2_cosine_dist = []
    tf_idf_1_2_word_app = []
    tf_idf_1_2_matching = []

    word2vec_cosine_dist = []

    for i in range(len(df)):
        contents = []
        for context in [df['ctx1_content_token'].iloc[i], df['ctx2_content_token'].iloc[i], df['ctx3_content_token'].iloc[i]]:
            if isinstance(context, list):
                contents.append(context)

        tf_idf_1_1_cosine_dist.append(tfidf_1_1.cosine_dist(df['content_token'].iloc[i], contents))
        tf_idf_1_1_word_app.append(tfidf_1_1.word_appearance(df['content_token'].iloc[i], contents))
        tf_idf_1_1_matching.append(tfidf_1_1.matching_score(df['content_token'].iloc[i], contents))

        tf_idf_1_2_cosine_dist.append(tfidf_1_2.cosine_dist(df['content_token'].iloc[i], contents))
        tf_idf_1_2_word_app.append(tfidf_1_2.word_appearance(df['content_token'].iloc[i], contents))
        tf_idf_1_2_matching.append(tfidf_1_2.matching_score(df['content_token'].iloc[i], contents))
        
        word2vec_cosine_dist.append(word2vec.cosine_dist(df['content_token'].iloc[i], contents))

    df['tf_idf_1_1_cosine_dist'] = tf_idf_1_1_cosine_dist
    df['tf_idf_1_1_word_app'] = tf_idf_1_1_word_app
    df['tf_idf_1_1_matching'] = tf_idf_1_1_matching

    df['tf_idf_1_2_cosine_dist'] = tf_idf_1_2_cosine_dist
    df['tf_idf_1_2_word_app'] = tf_idf_1_2_word_app
    df['tf_idf_1_2_matching'] = tf_idf_1_2_matching

    df['word2vec_cosine_dist'] = word2vec_cosine_dist

    tf_idf_1_1_cosine_dist = np.array(tf_idf_1_1_cosine_dist)
    tf_idf_1_1_word_app = np.array(tf_idf_1_1_word_app)
    tf_idf_1_1_matching = np.array(tf_idf_1_1_matching)

    tf_idf_1_2_cosine_dist = np.array(tf_idf_1_2_cosine_dist)
    tf_idf_1_2_word_app = np.array(tf_idf_1_2_word_app)
    tf_idf_1_2_matching = np.array(tf_idf_1_2_matching)

    df['tf_idf_1_1_harmonic_mean'] = 3 / ((1/tf_idf_1_1_cosine_dist) + (1/tf_idf_1_1_word_app) + (1/tf_idf_1_1_matching))
    df['tf_idf_1_2_harmonic_mean'] = 3 / ((1/tf_idf_1_2_cosine_dist) + (1/tf_idf_1_2_word_app) + (1/tf_idf_1_2_matching))
    return df
    
def jsd(p, q, base=np.e):
    '''Jenson-Shanon Distance
    Reference: https://stackoverflow.com/questions/20302636/js-divergence-between-two-discrete-probability-distributions-of-unequal-length
    '''
    if len(p) > len(q):
        p = np.random.choice(p, len(q)) # random.choice make same length to p/q
    elif len(q) > len(p):
        q = np.random.choice(q, len(p))
    p, q = np.asarray(p), np.asarray(q)
    
    return jensenshannon(p, q)

def boxplot_feats(df):
    feats = ['content_' + k for k in ['div_NOUN_sum', 'div_NOUN_percent', 'div_VERB_sum', 'div_VERB_percent', 'div_ADJ_sum', 'div_ADJ_percent', 'div_ADV_sum', 'div_ADV_percent', 'div_LEX_sum', 'div_LEX_percent', 'div_CONT_sum', 'div_CONT_percent', 'div_FUNC_sum', 'div_FUNC_percent', 'pron_FPS_sum', 'pron_FPS_percent', 'pron_FPP_sum', 'pron_FPP_percent', 'pron_STP_sum', 'pron_STP_percent', 'quant_NOUN_sum', 'quant_NOUN_percent', 'quant_VERB_sum', 'quant_VERB_percent', 'quant_ADJ_sum', 'quant_ADJ_percent', 'quant_ADV_sum', 'quant_ADV_percent', 'quant_PRON_sum', 'quant_PRON_percent', 'quant_DET_sum', 'quant_DET_percent', 'quant_NUM_sum', 'quant_NUM_percent', 'quant_PUNCT_sum', 'quant_PUNCT_percent', 'quant_SYM_sum', 'quant_SYM_percent', 'quant_PRP_sum', 'quant_PRP_percent', 'quant_PRP$_sum', 'quant_PRP$_percent', 'quant_WDT_sum', 'quant_WDT_percent', 'quant_CD_sum', 'quant_CD_percent', 'quant_VBD_sum', 'quant_VBD_percent', 'quant_STOP_sum', 'quant_STOP_percent', 'quant_LOW_sum', 'quant_LOW_percent', 'quant_UP_sum', 'quant_UP_percent', 'quant_NEG_sum', 'quant_NEG_percent', 'quant_QUOTE_sum', 'quant_NP_sum', 'quant_CHAR_sum', 'quant_WORD_sum', 'quant_SENT_sum', 'quant_SYLL_sum', 'senti_!_sum', 'senti_!_percent', 'senti_?_sum', 'senti_?_percent', 'senti_CAPS_sum', 'senti_CAPS_percent', 'senti_POL_sum', 'senti_SUBJ_sum', 'avg_chars_per_word_sum', 'avg_words_per_sent_sum', 'avg_claus_per_sent_sum', 'avg_puncts_per_sent_sum', 'med_st_ALL_sum', 'med_st_NP_sum']]
    read_feats = ['content_' + k for k in ['read_gunning-fog_sum', 'read_coleman-liau_sum', 'read_dale-chall_sum', 'read_flesch-kincaid_sum', 'read_linsear-write_sum', 'read_spache_sum', 'read_automatic_sum', 'read_flesch_sum']]
    total_feats = feats + read_feats

    new_dict = {}
    feat_dict = dict(np.floor(np.log10(df[total_feats].mean())))
    for k, v in feat_dict.items():
        new_dict[v] = new_dict.get(v, [])
        new_dict[v].append(k)

    fig, axes = plt.subplots(nrows=2, ncols= 4, figsize=(12, 8))
    ax = axes.ravel()

    for i, k in enumerate(sorted(new_dict.keys())):
        ax[i].boxplot(df[new_dict[k]], showfliers=False, vert=False)
        ax[i].set_title(f'$10^{{{int(k)}}}$')
        ax[i].set_yticklabels([i[8:] for i in new_dict[k]])

    plt.suptitle('Boxplot of Features by Scale of Powers of 10')
    fig.tight_layout()
    plt.show()

def plot_word_count(df, str_type_):
    # https://stackoverflow.com/questions/16180946/drawing-average-line-in-histogram-matplotlib
    plt.hist(df)
    plt.title(f'{str_type_} Word Count Histogram')
    plt.axvline(df.mean(), color='k', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(df.mean()*1.1, max_ylim*0.9, 'Mean: {:.2f}'.format(df.mean()))
    plt.legend(title='\n'.join(str(df.describe().round(2)).split('\n')[:-1]))
    plt.show()

def plot_word_cloud(df):
    # Without stop words
    word_cloud = WordCloud(width=800, height=800, background_color='white', stopwords=STOPWORDS).generate(" ".join(df['content']))

    plt.figure(figsize=(6,6))
    plt.imshow(word_cloud)
    plt.axis('off') 
    plt.tight_layout()
    plt.show()

def calculate_p(feats, df_feats, label):
    # Get all the features to feed into the two models
    f_stats, p_values = f_classif(df_feats, label)
    
    # ANOVA
    return sorted(list(zip(feats, p_values)), key=lambda x: x[1])

def plot_p_values_table(unsorted_p):
    sorted_p = sorted(unsorted_p, key=lambda x: x[1])
    fig, ax = plt.subplots()

    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    table = ax.table(cellText=[(x[0][8:], np.format_float_scientific(x[1], precision=2, exp_digits=2)) for x in sorted_p], colLabels=['category', 'p-value'], loc='center')

    for x in np.where(np.array([x[1] for x in sorted_p]) >= 0.05)[0]:
        table[(x + 1, 0)].set_facecolor("#56b5fd")
        table[(x + 1, 1)].set_facecolor("#56b5fd")

    fig.tight_layout()

    plt.show()

def plot_p_values(df):
    feats = ['content_' + k for k in ['div_NOUN_sum', 'div_NOUN_percent', 'div_VERB_sum', 'div_VERB_percent', 'div_ADJ_sum', 'div_ADJ_percent', 'div_ADV_sum', 'div_ADV_percent', 'div_LEX_sum', 'div_LEX_percent', 'div_CONT_sum', 'div_CONT_percent', 'div_FUNC_sum', 'div_FUNC_percent', 'pron_FPS_sum', 'pron_FPS_percent', 'pron_FPP_sum', 'pron_FPP_percent', 'pron_STP_sum', 'pron_STP_percent', 'quant_NOUN_sum', 'quant_NOUN_percent', 'quant_VERB_sum', 'quant_VERB_percent', 'quant_ADJ_sum', 'quant_ADJ_percent', 'quant_ADV_sum', 'quant_ADV_percent', 'quant_PRON_sum', 'quant_PRON_percent', 'quant_DET_sum', 'quant_DET_percent', 'quant_NUM_sum', 'quant_NUM_percent', 'quant_PUNCT_sum', 'quant_PUNCT_percent', 'quant_SYM_sum', 'quant_SYM_percent', 'quant_PRP_sum', 'quant_PRP_percent', 'quant_PRP$_sum', 'quant_PRP$_percent', 'quant_WDT_sum', 'quant_WDT_percent', 'quant_CD_sum', 'quant_CD_percent', 'quant_VBD_sum', 'quant_VBD_percent', 'quant_STOP_sum', 'quant_STOP_percent', 'quant_LOW_sum', 'quant_LOW_percent', 'quant_UP_sum', 'quant_UP_percent', 'quant_NEG_sum', 'quant_NEG_percent', 'quant_QUOTE_sum', 'quant_NP_sum', 'quant_CHAR_sum', 'quant_WORD_sum', 'quant_SENT_sum', 'quant_SYLL_sum', 'senti_!_sum', 'senti_!_percent', 'senti_?_sum', 'senti_?_percent', 'senti_CAPS_sum', 'senti_CAPS_percent', 'senti_POL_sum', 'senti_SUBJ_sum', 'avg_chars_per_word_sum', 'avg_words_per_sent_sum', 'avg_claus_per_sent_sum', 'avg_puncts_per_sent_sum', 'med_st_ALL_sum', 'med_st_NP_sum']]
    non_read_p_vals_sorted = calculate_p(feats, df[feats], df.label) 

    read_feats = ['content_' + k for k in ['read_gunning-fog_sum', 'read_coleman-liau_sum', 'read_dale-chall_sum', 'read_flesch-kincaid_sum', 'read_linsear-write_sum', 'read_spache_sum', 'read_automatic_sum', 'read_flesch_sum']]
    read_df = df[read_feats + ['label']]
    read_df.dropna()
    read_p_vals_sorted = calculate_p(read_feats, read_df[read_feats], read_df.label)
    sorted_p = sorted(non_read_p_vals_sorted + read_p_vals_sorted, key=lambda x: x[1])

    fig, axes = plt.subplots(ncols=2, figsize=(10, 8), sharex=True, sharey=True)
    fig.patch.set_visible(False)

    ax = axes.ravel()
    ax[0].axis('off')
    ax[0].axis('tight')
    ax[1].axis('off')
    ax[1].axis('tight')

    ax[0].table(cellText=[(x[0][8:], np.format_float_scientific(x[1], precision=2, exp_digits=2)) for x in sorted_p[:len(sorted_p)//2]], 
                cellColours=[('white', 'white') if x[1] < 0.05 else ('lightsteelblue', 'lightsteelblue') for x in sorted_p[:len(sorted_p)//2]],
                colLabels=['category', 'p-value'], loc='center')
    ax[1].table(cellText=[(x[0][8:], np.format_float_scientific(x[1], precision=2, exp_digits=2)) for x in sorted_p[len(sorted_p)//2:]], 
                cellColours=[('white', 'white') if x[1] < 0.05 else ('lightsteelblue', 'lightsteelblue') for x in sorted_p[len(sorted_p)//2:]],
                colLabels=['category', 'p-value'], loc='center')

    fig.suptitle('Features sorted by p-values', fontsize='xx-large')
    handles = [Rectangle((0, 0), 1, 1, color=c, ec="k") for c in ['white', 'lightsteelblue']]
    fig.legend(handles, ['< 0.05', '>= 0.05'], loc='upper right', bbox_to_anchor=(0.49, 0.45, 0.5, 0.55), fontsize='medium')
    fig.tight_layout()
    plt.show()
    return sorted_p

def plot_p_values_with_correlation(sorted_p, selected_feats):
    fig, axes = plt.subplots(ncols=2, figsize=(10, 8), sharex=True, sharey=True)
    fig.patch.set_visible(False)

    ax = axes.ravel()
    ax[0].axis('off')
    ax[0].axis('tight')
    ax[1].axis('off')
    ax[1].axis('tight')
    
    def cellText(i):
        if i==0:
            cells = sorted_p[:len(sorted_p)//2]
        else:            
            cells = sorted_p[len(sorted_p)//2:]
        out = []
        for x in sorted_p[:len(sorted_p)//2]:
            out.append((x[0][8:], np.format_float_scientific(x[1], precision=2, exp_digits=2)))
        return out
    
    def cellColours(i):
        if i==0:
            cells = sorted_p[:len(sorted_p)//2]
        else:            
            cells = sorted_p[len(sorted_p)//2:]
        out = []
        for x in cells:
            if x[0] in selected_feats:
                out.append(('white', 'white'))
            elif x[1] < 0.05:
                out.append(('lightcoral', 'lightcoral'))
            else:
                out.append(('lightsteelblue', 'lightsteelblue'))
        return out

    for i in range(2):
        ax[i].table(cellText=cellText(i), cellColours=cellColours(i), colLabels=['category', 'p-value'], loc='center')

    fig.suptitle('Features sorted by p-values', fontsize='xx-large')
    handles = [Rectangle((0, 0), 1, 1, color=c, ec="k") for c in ['white', 'lightsteelblue', 'lightcoral']]
    fig.legend(handles, ['< 0.05', '>= 0.05', 'Correlated'], loc='upper right', bbox_to_anchor=(0.49, 0.45, 0.5, 0.6), fontsize='medium')
    fig.tight_layout()
    plt.show()

def remove_correlated_feats(corr_matrx, p_values_sorted):
    upper_tri = corr_matrx.where(np.triu(np.ones(corr_matrx.shape),k=1).astype(np.bool))

    dict_ = dict(enumerate([i[0] for i in p_values_sorted]))
    new_dict = {}
    for k, v in dict_.items():
        new_dict[v] = int(k)

    to_iterate_l = sorted(list(upper_tri.index), key=lambda x: new_dict[x])
    graph_dict = {}

    to_skip = set()
    for row in to_iterate_l:
        if row in to_skip:
            continue
        for col in to_iterate_l:
            cell = upper_tri.loc[row][col]
            if cell > 0.95:
                graph_dict[row] = graph_dict.get(row, [])
                graph_dict[row].append(col)
                to_skip.add(col)

        # Solitary ones
        if graph_dict.get(row, []) == []:
            graph_dict[row] = []
        
    return graph_dict
    
def group_by_key(graph_dict):
    final_dict = {}
    for col in graph_dict.keys():
        k = col.split('_')[1]
        final_dict[k] = final_dict.get(k, [])
        final_dict[k].append(col)
    return final_dict

def plot_corr_heat_map(df, sorted_p):
    # https://lifewithdata.com/2022/03/13/how-to-remove-highly-correlated-features-from-a-dataset/
    # https://www.projectpro.io/recipes/drop-out-highly-correlated-features-in-python
    # create correlation  matrix
    low_p_vals = [y[0] for y in filter(lambda x: x[1] < 0.05, sorted_p)]
    corr_matrx = df[low_p_vals].corr().abs()
    disp_corr_matrx = corr_matrx.rename(columns={item:item[8:] for item in corr_matrx.columns})
    disp_corr_matrx = disp_corr_matrx.rename(index={item:item[8:] for item in corr_matrx.index})
    sns.heatmap(disp_corr_matrx)
    plt.title('Pearson Correlation Matrix')
    plt.show()
    
    return corr_matrx

def plot_method_comparison(df):
    colors = ["red", "green"]
    labels = ["Fake", "Real"]

    methods = ['word2vec_cosine_dist', 'tf_idf_1_1_cosine_dist', 'tf_idf_1_1_word_app', 'tf_idf_1_1_matching', 'tf_idf_1_2_cosine_dist', 'tf_idf_1_2_word_app', 'tf_idf_1_2_matching', 'tf_idf_1_1_harmonic_mean', 'tf_idf_1_2_harmonic_mean']
    f_stats, p_values = f_classif(df[methods], df.label)

    def plot_given_method(ax_, df, p_val, method, method_name):
        _, bins, _ = ax_.hist(df[df.label == 0][method], bins=20, color = colors[0])
        _ = ax_.hist(df[df.label == 1][method], bins=bins, alpha = 0.5, color = colors[1])

        # Use Jensen-Shannon Distance
        dist = jsd(df[df.label == 0][method], df[df.label == 1][method])
        delta_mu = abs(df[df.label == 0][method].mean() - df[df.label == 1][method].mean())

        ax_.set_title(f"{method_name}, Δµ: {delta_mu:.2f}, JSD: {dist:.2f}, p: {np.format_float_scientific(p_val, precision=2, exp_digits=2)}")

    fig, axes = plt.subplots(ncols=2, nrows=5, figsize=(12, 10), sharex=True, sharey=True)
    ax = axes.ravel()
    ax[1].set_visible(False)

    plot_given_method(ax[0], df, p_values[0], 'word2vec_cosine_dist', 'Word2Vec Cosine Dist')
    plot_given_method(ax[2], df, p_values[1], 'tf_idf_1_1_cosine_dist', 'TF-IDF (1-1) Cosine Dist')
    plot_given_method(ax[3], df, p_values[2], 'tf_idf_1_2_cosine_dist', 'TF-IDF (1-2) Cosine Dist')
    plot_given_method(ax[4], df, p_values[3], 'tf_idf_1_1_word_app', 'TF-IDF (1-1) Word App')
    plot_given_method(ax[5], df, p_values[4], 'tf_idf_1_2_word_app', 'TF-IDF (1-2) Word App')
    plot_given_method(ax[6], df, p_values[5], 'tf_idf_1_1_matching', 'TF-IDF (1-1) Matching Score')
    plot_given_method(ax[7], df, p_values[6], 'tf_idf_1_2_matching', 'TF-IDF (1-2) Matching Score')
    plot_given_method(ax[8], df, p_values[7], 'tf_idf_1_1_harmonic_mean', 'TF-IDF (1-1) Harmonic Mean')
    plot_given_method(ax[9], df, p_values[8], 'tf_idf_1_2_harmonic_mean', 'TF-IDF (1-2) Harmonic Mean')

    handles = [Rectangle((0, 0), 1, 1, color=c, ec="k") for c in colors]
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.49, 0.45, 0.5, 0.5), fontsize='xx-large')

    fig.suptitle('Method Comparisons', fontsize='xx-large')
    fig.tight_layout()
    plt.show()