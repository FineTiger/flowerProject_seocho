import cv2
import pymysql
from flask import Flask, render_template, request, jsonify
from keras.models import load_model
import numpy as np
from werkzeug.utils import secure_filename

db = pymysql.connect(host='localhost', port=3306, user='root', passwd='1234', db='ai_college', charset="utf8")
cur = db.cursor()  # 커서 클래스 호출

app = Flask(__name__)


@app.route("/")
@app.route("/index")
def index():
    return render_template('index.html')


@app.route("/input-image", methods=["GET", "POST"])
def input_image():
    if request.method == "POST":
        print(request.files)
        try:
            if 'input-image' in request.files:
                imageFile = request.files['input-image']
                savePath = "static/saved_file/"
                imageFile.save(savePath + secure_filename(imageFile.filename))
        except Exception as e:
            print(e)
        return jsonify("Ok")


@app.route("/image", methods=["POST"])
def input_image2():
    file = request.files["in_image"]
    print(file)
    # file2 = request.form["in_image"]
    # print(file2)
    # f_name = file.filename
    # file.save('static/saved_file/' + secure_filename(f_name))

    return "확인"


@app.route("/analyse_result", methods=["GET"])
def analyse_image():
    # 이미지 파일명 가져오기
    f_name = request.args.get("img_name")
    print(f_name)

    # 전처리
    IMG_SIZE = 150
    img = cv2.imread('static/saved_file/' + secure_filename(f_name), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = np.array(img)
    X = []
    X.append(img)
    X = np.array(X)
    X = X / 255

    # 입력 받은 이미지 예측
    expred = model.predict(X)
    expred_digits = np.argmax(expred, axis=1)

    # 분류
    answer = expred_digits
    if answer == 0:
        ans = "백합_시베리아"
    if answer == 1:
        ans = "안개_오버타임"
    if answer == 2:
        ans = "용담_용담"
    if answer == 3:
        ans = "카네이션_빨간카네이션"
    if answer == 4:
        ans = "해바라기_해바라기"

    # 문자열 스플릿
    fl_item, fl_type = str(ans).split('_')

    cur.execute(
        # "SELECT * FROM realtime_flower where poomname = '%s' and goodname = '%s' order by qty" % (
        # fl_item, fl_type))  # 모델 결과값에 따라 쿼리문 작성하는 부분
        "SELECT poomname, goodname, lvname, qty, round(max(cost))"
        " FROM realtime_flower where poomname = '%s' and goodname = '%s' group by lvname;" % (
        fl_item, fl_type))  # 모델 결과값에 따라 쿼리문 작성하는 부분
    rows = cur.fetchall()  # 데이터저장
    print(rows)

    # 딕셔너리에 담아서 json 형태로 변환하여 전송
    flowers_dict = []
    for row in rows:
        flowers_dict.append({
            "poomname": row[0],
            "goodname": row[1],
            "lvname": row[2],
            "qty": row[3],
            "cost": row[4]
        })

    return jsonify(flowers_dict)


if __name__ == '__main__':
    # ml/project_code_final.py 선 실행 후 생성
    model = load_model('./ml/model.h5')

    # Flask 서비스 스타트
    app.run(debug=True)
