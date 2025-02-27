'''
今回は5つの機能を考えました。
1, TFIDFによる検索機能です。内容は、検索した単語が歌詞の中の重要度の高い位置付けとなっている楽曲を表示します。
また、検索ワードと関連性の高い３つの単語も共に検索します。メリットは、例えば夏と検索した時に、夏だけでなく海などの
単語も検索されより幅広く曲を検索できます。夏っぽくても夏という歌詞が入っているとは限らないのでこちらを搭載することを
考えました。今回は歌詞内の単語の検索しかできないので、今後は歌詞からも同義語を照合対象にするなどして、より柔軟性を
高くしていきたいと思います。

2, 歌詞の類似度検索機能です。各歌詞のベクトルをFAISSで演算を行い似た歌詞を持つと判定されたものを鑑賞できるよう
記載します。上下どの機能を使って移ったページからも、曲名を押すことでこの機能を使ったページに移るという仕組みになっています。

3, 音楽再生、歌詞表示機能です。仮にアプリに載っていなかった楽曲があった場合に自分でファイルを載せて再生して、音声認識で
歌詞表示する機能です（こちらは何かメリットというよりただ音声認識のコードを書いてみたかっただけです）。

4, 再生履歴のデータをもとに、ユーザー好みの曲を分類してレコメンドする機能です。基本は楽曲と歌詞の情報をもとに分類しており、
アプリを操作するときにレコメンドのボタンを一度押すとどのようなグループ分けになっているのかを数値で見ることができるように
なっています。数値は、グループ分けされた楽曲の項目ごとの値の平均値です。メリットは曲調と歌詞両方の観点での客観的なデータを
ユーザーが自ら見て、適切な楽曲グループを選択して聴くことができる点だと考えます。

5, 再生履歴データをもとに、次にどのような楽曲を聴きたいかを時系列で追っていくという考えをもとに推測してレコメンドする機能です。
こちらはまとまった時間で音楽を聴きたいと思った時に使える機能だと考えます。メリットはユーザーがわざわざ選択しなくても自動で
気分にあった曲を聴き続蹴ることが期待できる点です。今回は時間が足りず、単純に時系列で特徴量を分析するということで終わって
しまいました。今後の施策として考えられることは、複数のユーザーのデータをロードして、似た傾向のユーザの曲の選定を基準に
曲そのものをラベルとしてレコメンドする、深層強化学習やGNNによって楽曲とユーザの関係を学習して長期的に反映することなどが
考えらえます。
'''
import csv
import ctypes
import os
import pickle
import threading
import faiss
import librosa
import numpy as np
import pandas as pd
import scipy.signal
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
import whisper
from flask import Flask
from flask import request, jsonify, render_template
from scipy.sparse import csr_matrix, hstack
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sudachipy import dictionary
from sudachipy import tokenizer

app = Flask(__name__)

# 履歴を保存するファイル
HISTORY_FILE = 'history.csv'
MAX_HISTORY_SIZE = 100

# 履歴を保存する関数
def save_history(row):
    # 履歴を読み込む→リスト化
    try:
        with open(HISTORY_FILE, 'r', newline='') as file:
            reader = csv.reader(file)
            history = list(reader)
    except FileNotFoundError:
        history = []

    # 履歴を追加
    history.append(row)

    # 履歴がMAX_HISTORY_SIZEを超えた場合、古いものから削除
    if len(history) > MAX_HISTORY_SIZE:
        history = history[-MAX_HISTORY_SIZE:]

    # 履歴を保存
    with open(HISTORY_FILE, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(history)

# 歌詞データCSV
df = pd.read_csv('comdata.csv') #faiss用
dfe = df.copy() #cluster用↓これらは固有のものとなるので削除
dfe = dfe.drop(columns=['曲名', '歌い出し', 'name', 'album', 'artist', 'popularity'])

#javascriptで履歴のコードを書いた
@app.route('/save_history/<int:song_index>', methods=['POST'])
def save_history_endpoint(song_index):
    # 履歴を保存
    save_history(dfe.iloc[song_index].to_list())
    return jsonify({'status': 'success'})

#共起行列データ
dfc = pd.read_csv('co_occurrence_matrix_lylics.csv', header=None, names=['Word1', 'Word2', 'Count'])
dfc['Count'] = pd.to_numeric(dfc['Count'], errors='coerce')

# Sudachiの形態素解析器を設定
tokenizer_obj = dictionary.Dictionary().create()

def sudachi_tokenizer(text):
    # テキストを形態素解析し、基本形をリストとして返す→.dictionary_form()
    tokens = [m for m in tokenizer_obj.tokenize(text, tokenizer.Tokenizer.SplitMode.C)]
    filtered_tokens = [m.dictionary_form() for m in tokens if m.part_of_speech()[0] in ['名詞', '形容詞', '動詞']]
    return filtered_tokens

# TF-IDFベクトル化器を作成、形態素解析器はsudachi、基本形返す
vectorizer = TfidfVectorizer(tokenizer=sudachi_tokenizer)

# 歌詞データに対して形態素解析を行い、TF-IDF行列を作成
tfidf_matrix = vectorizer.fit_transform(df['歌い出し'].apply(lambda x: ' '.join(sudachi_tokenizer(x))))
#0でない → 各ドキュメントにおいて何種類の単語が存在するか
non_zero_counts = np.count_nonzero(tfidf_matrix.toarray(), axis=1)
dfe['words'] = pd.DataFrame(non_zero_counts, columns=['NonZeroCount'])

# vecsをNumpy→ノルムで割って正規化→faiss index作成→ベクトルをインデックスに追加
with open('vecs.pkl', 'rb') as f:
    vecs = pickle.load(f)
vecs_array = np.array(vecs).astype("float32")
vecs_array /= np.linalg.norm(vecs_array, axis=1)[:, np.newaxis]
index_f = faiss.IndexFlatIP(768)  # BERT(SimCSE)は768次元
index_f.add(vecs_array)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        search_word = request.form.get('search_word', None)
        if search_word:
            # 検索機能のコードをここに追加
            title_matches = df[df['曲名'] == search_word]
            title_results = [(row['曲名'], row['artist'], row.name, 'dummy_audio.wav', '歌詞') for _, row in title_matches.iterrows()]

            filtered_df = dfc[dfc['Word1'] == search_word]
            top3_rows = filtered_df.nlargest(3, 'Count')
            values_d = top3_rows['Word2'].tolist()
            values_d.append(search_word)
            search_words = values_d

            tfidf_values_combined = []
            for s in search_words:
                word_index = vectorizer.vocabulary_.get(s)
                if word_index is not None:
                    word_tfidf_values = tfidf_matrix.getcol(word_index).toarray().flatten()
                    word_tfidf_sparse = csr_matrix(word_tfidf_values).T
                    tfidf_values_combined.append(word_tfidf_sparse)

            if tfidf_values_combined:
                tfidf_combined_matrix = hstack(tfidf_values_combined)
                top10 = (tfidf_combined_matrix.sum(axis=1).A1 / dfe['words']).argsort()[-10:][::-1]
                results = []
                for idx in top10:
                    song_name = df.iloc[idx]['曲名']
                    if song_name != search_word:
                        results.append((song_name, df.iloc[idx]['artist'], idx, 'dummy_audio.wav'))
            else:
                results = []

            final_results = title_results + results
            return render_template('resultsaudio.html', search_word=search_word, results=final_results)

    return render_template('indexhightext.html')

# ここから音声認識
model = whisper.load_model("small.en")
dll = ctypes.CDLL("./stereophonic_sound_mixed/x64/Debug/stereophonic_sound_mixed.dll")

# DLL関数の引数と返り値の型を指定
dll.load_audio.argtypes = [ctypes.c_char_p]
dll.load_audio.restype = None
dll.play_audio.restype = None
dll.stop_audio.restype = None
dll.cleanup_audio.restype = None

recognized_text = []
streaming = False
file_path = ""

def preprocess_audio(audio_array, sampling_rate):
    audio_array = np.float32(audio_array)  # データ型をfloat32に変換
    sos = scipy.signal.butter(10, [30, 3000], btype='bandpass', fs=sampling_rate, output='sos')
    filtered_waveform = scipy.signal.sosfilt(sos, audio_array)
    return filtered_waveform

def clean_text(text):
    return ' '.join(text.split())

def remove_duplicates(text_list):
    return list(dict.fromkeys(text_list))

def process_audio(file_path):
    global recognized_text, streaming
    recognized_text.clear()
    try:
        # ファイルパスの確認
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"ファイルが存在しません: {file_path}")

        audio, sample_rate = librosa.load(file_path, sr=16000, mono=True)
        audio = preprocess_audio(audio, sample_rate)  # フィルタリングされた音声データを取得

        # 一時ファイルにフィルタリング後のオーディオデータを書き込む
        temp_file_path = "filtered_temp.wav"
        sf.write(temp_file_path, audio, sample_rate)

        # Whisperモデルで音声ファイルを処理
        result = model.transcribe(temp_file_path)
        text = clean_text(result["text"])
        recognized_text.extend(remove_duplicates(text.split("\n")))
        print(f"Recognized text: {recognized_text}")
        os.remove(temp_file_path)
    except Exception as e:
        print(f"音声処理中にエラーが発生しました: {e}")
    finally:
        streaming = False  # 音声処理が終了したらstreamingをFalseに設定
        print("Stream ended. Final recognized text:", recognized_text)

@app.route('/lyrics')
def lyrics_page():
    return render_template('lyrics.html')

@app.route('/recognized-text', methods=['GET'])
def get_recognized_text():
    global streaming
    if not streaming:
        return jsonify(recognized_text)
    else:
        return jsonify([])

@app.route('/upload', methods=['POST'])
def upload_file():
    global recognized_text, streaming, file_path
    recognized_text.clear()
    streaming = True
    if 'file' not in request.files:
        return 'ファイルがありません。', 400
    file = request.files['file']
    if file.filename == '':
        return 'ファイル名が無効です。', 400
    if file:
        file_path = os.path.join('./uploaded_files', file.filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        file.save(file_path)

        try:
            file_path_encoded = file_path.encode('utf-8')
            dll.load_audio(file_path_encoded)
        except Exception as e:
            return f"DLL呼び出し中にエラーが発生しました: {str(e)}", 500

        threading.Thread(target=process_audio, args=(file_path,)).start()
        return jsonify({"message": "ファイルが正常にアップロードされ、音声認識を開始しました。"})

@app.route('/play', methods=['POST'])
def play_audio():
    global streaming, file_path
    try:
        streaming = True
        threading.Thread(target=dll.play_audio).start()
        return jsonify({"message": "音声を再生しました。"})
    except Exception as e:
        return jsonify({"message": f"音声再生中にエラーが発生しました: {str(e)}"}), 500

@app.route('/stop', methods=['POST'])
def stop_audio():
    global streaming
    try:
        streaming = False
        dll.stop_audio()
        return jsonify({"message": "音声を停止しました。"})
    except Exception as e:
        return jsonify({"message": f"音声停止中にエラーが発生しました: {str(e)}"}), 500

@app.route('/clear-lyrics', methods=['POST'])
def clear_lyrics():
    global recognized_text
    recognized_text.clear()
    return jsonify({"message": "歌詞がクリアされました。"})


@app.route('/song/<int:song_index>', methods=['GET'])
# song.htmlのindex→song関数のsong_index→（結果的に）song関数のindex
def song(song_index):
    # 検索対象ベクトルの取得
    query_vec = vecs_array[song_index].reshape(1, -1)

    # 検索 (類似度上位10件を取得、D：類似度スコア、I：インデックス)
    k = 11
    D, I = index_f.search(query_vec, k)
    D, I = D[0][1:], I[0][1:]  # 自分自身を除外

    # 結果を格納するリスト(<int:song_index>html、song関数下部では結果的にindex)
    results = [(df.iloc[index]['曲名'], df.iloc[index]['artist'], index, 'dummy_audio.wav') for index in I]
    save_history(dfe.iloc[song_index].to_list()) # 履歴を保存

    return render_template('song.html', song_title=df.iloc[song_index]['曲名'], results=results)

# ここからクラスタリング
def cluster_search():
    # 履歴を読み込む
    try:
        with open(HISTORY_FILE, 'r', newline='') as file:
            reader = csv.reader(file)
            history = list(reader)
    except FileNotFoundError:
        return [], []

    # 履歴データを数値に変換
    numeric_history_df = pd.DataFrame(history, columns=dfe.columns)
    numeric_history_df = numeric_history_df.apply(pd.to_numeric, errors='coerce')  # 数値変換できないものをNaNに変換

    # 欠損値の処理（必要に応じて）
    numeric_history_df = numeric_history_df.dropna()

    # 特徴量のスケーリング
    scaler = StandardScaler()
    numeric_columns = numeric_history_df.columns
    scaled_features = scaler.fit_transform(numeric_history_df)
    scaled_numeric_history_df = pd.DataFrame(scaled_features, columns=numeric_columns, index=numeric_history_df.index)

    # PCAによる次元削減の導入
    pca = PCA(n_components=0.9)  # 分散の90%を保持する主成分数を自動選択
    pca_features = pca.fit_transform(scaled_numeric_history_df)
    pca_numeric_history_df = pd.DataFrame(pca_features, index=scaled_numeric_history_df.index)

    # エルボー法の実行
    sse = []
    for k in range(1, min(7, len(pca_numeric_history_df) + 1)):  # クラスタ数をサンプル数以下に制限
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(pca_numeric_history_df)
        sse.append(kmeans.inertia_)  # inertia_=SSE（残差平方和）

    # 最適なクラスタ数を自動的に選択
    def optimal_number_of_clusters(sse):
        x = range(1, len(sse) + 1)
        y = sse
        max_slope = 0
        optimal_k = 1
        for i in range(1, len(x) - 1):
            if y[i+1] != y[i]:
                slope = (y[i] - y[i-1]) / (y[i+1] - y[i])  # xの差は必ず1

                if slope > max_slope:
                    max_slope = slope
                    optimal_k = i + 1
        return min(optimal_k, 6)  # クラスタ数を最大6つに制限

    optimal_clusters = optimal_number_of_clusters(sse)
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=0)
    clusters = kmeans.fit_predict(pca_numeric_history_df)

    # クラスタごとの平均値を計算（PCA空間上）
    cluster_means = pd.DataFrame(pca_numeric_history_df).groupby(clusters).mean()

    # PCA空間のクラスタ平均を元の特徴量空間に逆変換
    cluster_means_original_pca = pca.inverse_transform(cluster_means)

    # スケーリングを元に戻す
    cluster_means_original_scale = scaler.inverse_transform(cluster_means_original_pca)
    cluster_means_original_scale = pd.DataFrame(cluster_means_original_scale, columns=numeric_columns)

    # クラスタごとの平均特徴量とデータフレーム内の曲の特徴量とのコサイン類似度を計算
    cluster_similarities = []
    for cluster_mean in cluster_means_original_scale.values:
        similarities = cosine_similarity(dfe.values, [cluster_mean])
        top_10_indices = np.argsort(similarities.flatten())[-10:][::-1]
        top_10_titles = [(df.iloc[index]['曲名'], df.iloc[index]['artist'], index, 'dummy_audio.wav') for index in top_10_indices]
        cluster_similarities.append(top_10_titles)

    return cluster_means_original_scale, cluster_similarities


@app.route('/clusters', methods=['GET'])
def clusters():
    cluster_means, cluster_similarities = cluster_search()
    if len(cluster_means) == 0 or len(cluster_similarities) == 0:
        return render_template('clusters.html', message='履歴がありません')
    column_names = dfe.columns.tolist()
    similarities = cluster_similarities
    return render_template('clusters.html', cluster_means=cluster_means, similarities=similarities, column_names=column_names)


@app.route('/cluster_n/<int:cluster_id>')
def cluster_n(cluster_id):
    cluster_means, cluster_similarities = cluster_search()
    similarities = cluster_similarities[cluster_id]
    link_name = f"旋律の環 {cluster_id + 1}"
    return render_template('cluster_n.html', similarities=similarities, link_name=link_name)


# ここから時系列
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super(RNNModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# 時系列予測を行う関数
def predict_next_songs(model, history, n_predictions):
    model.eval()
    predictions = []
    input_seq = history.clone().detach().float() #バッチ　タイムステップ　特徴量
    with torch.no_grad(): #10データポイント分の推論
        for _ in range(n_predictions):
            yhat = model(input_seq) #バッチ　タイムステップ　特徴量 →　バッチ　特徴量 [1,22]
            predictions.append(yhat.squeeze(0).cpu().numpy()) #バッチ　特徴量 [1,22] →　特徴量 22
            #シーケンスの0番目のデータを削除して予測データを追加
            input_seq = torch.cat((input_seq[:, 1:, :], yhat.unsqueeze(0)), dim=1)  # シーケンスを更新
    return np.array(predictions)  # 2次元に変換

def timeseries_search():
    # 履歴を読み込む
    try:
        with open(HISTORY_FILE, 'r', newline='') as file:
            reader = csv.reader(file)
            history = list(reader)
    except FileNotFoundError:
        return []

    if len(history) < 21:
        return []

    history_df = pd.DataFrame(history, columns=dfe.columns)
    history_df = history_df.apply(pd.to_numeric)

    scaler = StandardScaler()
    scaler.fit(dfe)  # 全データでフィッティング
    scaled_history = scaler.transform(history_df)
    scaled_dfe = scaler.transform(dfe)

    pca = PCA(n_components=2)  # 次元数を2に削減
    pca.fit(scaled_dfe)  # 全データでフィッティング
    reduced_history = pca.transform(scaled_history)
    reduced_dfe = pca.transform(scaled_dfe)

    # NumPy配列をPyTorchテンソルに変換
    reduced_history_tensor = torch.tensor(reduced_history[-21:-2], dtype=torch.float32)
    reduced_history_tensor = reduced_history_tensor.unsqueeze(0)  # (1, 履歴数, 2)

    reduced_history_tensor_y = torch.tensor(reduced_history[-1], dtype=torch.float32)
    reduced_history_tensor_y = reduced_history_tensor_y.unsqueeze(0)  # (1, 履歴数, 2)

    model = RNNModel(input_size=2, hidden_size=50, output_size=2, num_layers=2, dropout=0.5)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        output = model(reduced_history_tensor)
        loss = criterion(output, reduced_history_tensor_y)
        loss.backward()
        optimizer.step()

    # レコメンド結果を取得
    recommendations = recommend_songs(model, reduced_history_tensor, 10, df, reduced_dfe)
    return recommendations

# 曲を推薦する関数
def recommend_songs(model, history, n_predictions, df, reduced_dfe):
    # 10データ分予測を行う
    predicted_songs = predict_next_songs(model, history, n_predictions)

    recommendations = []
    used_indices = set()

    for predicted in predicted_songs:
        # 1データごとにコサイン類似度を計算
        similarities = cosine_similarity([predicted], reduced_dfe)

        # 類似度の高い順にインデックスを取得
        sorted_indices = np.argsort(-similarities[0])

        # 重複があれば2番目の類似度の曲選択
        for index in sorted_indices:
            if index not in used_indices:
                used_indices.add(index)
                recommendations.append(
                    (df.iloc[index]['曲名'], df.iloc[index]['artist'], index, 'dummy_audio.wav')
                )
                break  # 最も類似度が高い曲を選択したらループを抜ける
            else:
                continue  # 重複があれば次のインデックスを確認

    return recommendations

#pythonでレンダリング→HTMLで受け取る→jsに関数が渡される→履歴保存
@app.route('/timeseries', methods=['GET'])
def timeseries():
    recommendations = timeseries_search()
    if not recommendations:
        return render_template('timeseries.html', message='履歴がありません')
    return render_template('timeseries.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
