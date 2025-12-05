
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from model import predict

st.sidebar.title("画像認識アプリ")
st.sidebar.write("オリジナルの画像認識モデルを使って何の画像かを判定します。")
st.sidebar.write("")

img_source = st.sidebar.radio("画像のソースを選択してください。",("画像をアップロード","カメラで撮影"))

if img_source == "画像をアップロード":
  img_file = st.sidebar.file_uploader("画像を選択してください。",type=["png","jpg","jpeg"])
elif img_source == "カメラで撮影":
  img_file = st.camera_input("カメラで撮影")

if img_file is not None:
  with st.spinner("推定中..."):
    img = Image.open(img_file)
    st.image(img, caption="対象の画像", width=480)
    st.write("")

    results = predict(img)

    st.subheader("判定結果")
    n_top = 3
    for result in results[:n_top]:
      st.write(str(round(result[2]*100,2)) + "%の確率で" + result[0] + "です。")

    pie_labels = [result[1] for result in results[:n_top]]
    pie_labels.append("others")
    pie_probs = [result[2] for result in results[:n_top]]
    pie_probs.append(sum([result[2] for result in results[n_top:]]))
    fig, ax = plt.subplots(figsize=(7,7))
    wedgeprops = {"width":0.3, "edgecolor":"white"}
    textprops = {"fontsize":6}
    ax.pie(
    pie_probs,
    labels=pie_labels,
    counterclock=False,
    startangle=90,
    textprops=textprops,
    autopct="%.1f%%",
    wedgeprops=wedgeprops,
    pctdistance=0.85,    # パーセンテージを外側へ
    labeldistance=1.10   # ラベルも外へ
    )

    st.pyplot(fig)

st.sidebar.write("")
st.sidebar.write("")

st.sidebar.caption("""
このアプリは、「Fashion-MNIST」を訓練データとして使っています。\n
Copyright (c) 2017 Zalando SE \n
Released under the MIT licence \n
http://github.com/zalandoresearch/fashion-mnist#license
""")
