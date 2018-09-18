#seleniumのwebdriverをインポート
from selenium import webdriver
 
#chromeを開く
chro = webdriver.Chrome()
 
#urlを指定する
chro.get("https://keras.io/ja/optimizers/")
 
#chrome閉じる
#chro.quit()

