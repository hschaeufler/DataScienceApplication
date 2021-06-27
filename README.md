# DataScienceApplication
Die Datenquelle ist der Covid-19 Datenhub des Robert Koch-Institut (RKI). Die Daten stehen unter der Open Data Datenlizenz Deutschland – Namensnennung – Version 2.0 zu Verfügung.

https://npgeo-corona-npgeo-de.hub.arcgis.com/datasets/dd4580c810204019a7b8eb3e0b329dd6_0

Weitere Infos zum Datensatz finden sich auf der Seite des RKIs bzw. im Ordner Data in der Readme-Datei.

## Setup
Als Erstes bitte Docker Desktop installieren.

### GIT LFS
Da es sich bei den Datensätzen um sehr große Dateien handelt, wird das Git Large File Storage-Plugin benötigt. Dieses ersetzt große Dateien durch einen Pointer und speichert die Dateien auf einem Remote-Server (wie Github.com). Das entsprechende Plugin kann von folgender URL heruntergeladen werden:

https://git-lfs.github.com/

### Klonen
Das Repository kann mittels folgendem Befehl geklont werden.
``
git clone https://github.com/hschaeufler/DataScienceApplication.git
``

### Jupyter-Umgebung
Um die Notebooks betreiben, bitte auch folgendes Repository klonen.
``
https://github.com/sturc/jupyter_with_yarn.git
``

Danach in den ``jupyter_with_yarn`` Ordner navigieren und dort in das .env File nach ``DS_BD_DIR=`` den Pfad zum Ordner das DataScienceApplication-Projekts eintragen.

Das File sollte dann etwa so ähnlich aussehen.
```
DS_BD_DIR=C:\workspaces\datascience\ds_bd\DataScienceApplication
```

Danach kann über ``docker compose up`` (innerhalb des Ordners jupyter_with_yarn) die Umgebung gestartet werden.

Um die Jypyter-Notebook-Umgebung zu starten, bitte die URL mit dem Access-Token, der Konsole entnehmen, oder folgende [Readme](https://github.com/sturc/jupyter_with_yarn) beachten.

### Deployment ins Hadoop-Cluster
Mit dem Notebook ```push_data_dir_to_hdfs.ipynb``` können die CSV-Files im HDFS des Hadoop-Cluster gespeichert werden.

Mit folgendem Befehl kann dann das Random-Forest-Python-Script (05_Deployment_RF.py) in das Hadoop-Cluster deployed und als Job submitet werden.
```
spark-submit --master yarn --deploy-mode client --conf spark.network.timeout=121s --conf spark.executor.heartbeatInterval=120s --conf spark.yarn.maxAppAttempts=2 work/DataScienceApplication/05_Deployment_RF.py
```
## License Information
Aus dem GitHub-Repository https://github.com/sturc/ds_bd wurden nachfolgende Files und Ordner (inkl. Inhalt) übernommen.
```
\binder
\hadoop-configs
\helpers
push_data_dir_to_hdfs.ipynb
requirements.txt
requirements_pyspark_notebook.txt
```
Die Inhalte des Repositorys, aus dem die entsprechenden Dateien übernommen wurden stehen unter der [APACHE 2.0 LICENSE](https://github.com/sturc/ds_bd/blob/master/LICENSE) zur verfügung.
