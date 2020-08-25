# ScrapyTutorial
# CYO道より飲み会アプリ


## 初期設定
* 開発プロジェクトをgit cloneする
``` bash
　git clone ssh://git@gitlab.avelio.jp:60122/CYOFindFoodApp/scrapytutorial.git 
```

* Dockerマシンを起動
　
```bash
　docker-machine start
```
　
　
* Dockerマシンを起動確認
　
```bash
　docker-machine ls
```

* Dockerコンテナをビルド、立ち上げる
``` bash
　docker-compose up
```

* 新しいターミナルを開き、Dockerコンテナが正常に起動したか確認
``` bash
　docker-compose ps 
```
（例）
```
           Name              Command   State   Ports
----------------------------------------------------
testenv_your-sevice-name_1   python3   Up


```

* コンテナに入る

``` bash
winpty docker exec -it testenv_your-sevice-name_1 bash
```

* コンテナに入った後、python実行、「Hello World」って表示されたら成功

``` bash
python test.py
```


