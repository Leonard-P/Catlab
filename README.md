# Catlab
### AI to identify missing cats

## 2024 update
The model.h5 file was re-exported in the native keras format ("catlab_model_2024") for compatibility with newer python and tensorflow versions. It was tested under
* Python 3.10.12
* Tensorflow 2.15.0
The jupyter notebook "Catlab2024.ipynb" gives an example of loading and using the model. It is intended to run on Google Colaboratory.


## Introduction (German)
Catlab ist gewissermaßen eine Gesichtserkennung für Katzen - nur dass der ganze Tierkörper in die Entscheidung einfließt. Der Hauptanwendungszweck ist die Identifizierung vermisster Katzen durch das Sortieren von Vermisstenmeldungen nach Ähnlichkeit zu einem Referenzbild, das ein potenzieller Finder einer Katze erstellt.
Aber auch für weitere Zwecke, wie Katzenklappen oder Futterautomaten mit Gesichtserkennung, kann die Beachtung des ganzen Körpers von Vorteil sein.

## Unser Ansatz
Zunächst schrieben wir für den Datensatz einen Crawler, um Bilder von Onlinetierbörsen wie Quoka zu downloaden. Außerdem verwendeten wir die API des Tieradoptionsportals Petfinder. So erhielten wir mehrere jeweils mehrere Bilder von derselben Katze, was Voraussetzung für das Training eines Siamese Networks ist - das die Grundlage unserer Modellarchitektur bildet (insgesamt enthielt der Datensatz ca. 60000 Bilder von 20000 verschiedenen Katzen). Mit YOLO erkannten wir Bilder, die nicht genau eine Katze zeigen, und sortierten wir diese aus. Die Testmenge bereinigten wir zudem per Hand. Um zu verhinden, dass das Netz irrelevante Merkmale, primär den Bildhintergrund, in die Entscheidung einfließen lässt, stellten wir die Katzen außerdem vom Hintergrund frei. Bezüglich der Architektur des Modells entschieden wir uns für ein Siamese Network, für dessen „Zwillinge“ wir die Convolutional Base des EfficientnetB0 und weitere eigene Schichten verwendeten. Durch diese Modellarchitektur können die Merkmale jedes Katzenbilds in einen 128-dimensionalen Vektor gespeichert werden. Der Abstand zweier Vektoren gibt die Ähnlichkeit zweier Katzen an. Mit den Vektorabständen kann man z.B. Katzen nach Ähnlichkeit sortieren oder vorhersagen, ob zwei Bilder die selbe Katze zeigen.

# Demonstration
Das Notebook ‘Demo.ipynb’ lässt sich einfach mit Google Colaboratory öffnen und ausführen (Es ist wichtig, alle Zellen von oben nach unten auszuführen). Im Notebook wird aus dem Dogs-vs-Cats-Datensatz ein zufälliges Referenzbild ausgewählt, zu dem unser Netz die ähnlichsten Bilder aus dem Datensatz auswählt. Durch erneutes Ausführen der letzten Zelle wird eine neue Sortierung generiert.

# Web-Demonstration
Zur Demonstration der Funktionalität des Neuronalen Netzes haben wir eine kleine Website programmiert. Sie lässt sich über [diesen Link](https://leonard-p.github.io) aufrufen.

# Verwendung
## Benötigte Software:
Wir empfehlen die Verwendung mit
* Python 3.7
* Tensorflow 2.1.x
* EfficientNet 1.1.1

Aber das Modell lässt sich auch mit so gut wie allen weiteren Sprachen verwenden, solange diese Tensorflow unterstützen.

## Preprocessing
Zur Vorbereitung der Bilder für das neuronale Netz sollten diese auf das Format 1:1 zugeschnitten werden und mit der Tensorflow-Funktion
```python
tensorflow.keras.applications.imagenet_utils.preprocess_input(IMAGE, mode="torch")
```
fertig vorverarbeitet werden. IMAGE ist ein numpy.array mit Werten von 0 bis 255. <br>

Für die Vorverarbeitung in anderen Sprachen befindet sich ein Beispiel für JavaScript im Repository unserer Website. Dieses ist derzeit noch in der Entwicklung und wird vermutlich ab Ende März veröffentlicht. 

## Input- / Outputformat
### Vektorabstand zweier Bilder berechnen
Zur Berechnung des Vektorabstandes eines Bilderpaars bzw. mehrerer Bilderpaare erwartet das Modell zwei Listen mit jeweils einem Bild der Paare; es erwartet also Eingaben der Form (2, BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3). IMAGE_SIZE ist die Auflösung der Bilder, für die wir 350x350 Pixel empfehlen (zum Training verwendeten wir 224x224 Pixel). Die Auflösung ist jedoch frei wählbar. <br>

Die Ausgabe ist eine Matrix an Gleitkommazahlen der Form (BATCH_SIZE, 1), die den Vektorabstand angibt. Ist der Abstand größer als 0.412, handelt es sich vermutlich um verschiedene Katzen.

### Einzelne Einbettungs-Vektoren berechnen
Um einen einzelnen Feature Vector für ein oder mehrere Bilder zu berechnen, benötigt man erst das Teilnetz des gesamten siamesischen Netzwerks. In Python erhält man dies mit
```python
base_model = model.layers[2]
```
Das base_model erwartet eine Liste mit Bildern der Form (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3). Als IMAGE_SIZE empfehlen wir für die Inferenz wieder 350x350 Pixel.
Die Ausgabe ist eine Matrix der Form (BATCH_SIZE, 128). Sie entspricht einer Liste an Einbettungs-Vektoren der Dimensionalität 128.

# Ergebnisse
Man kann den Abstand der Einbettungen zweier Bilder dazu nutzen, zu klassifizieren, ob auf zwei Bildern die selbe Katze abgebildet ist. Wir empfehlen hierzu einen Schwellwert von 0.412. Das heißt, wir haben festgelegt, dass ein Abstand von kleiner als 0.412 bedeutet, dass zwei Bilder die selbe Katze zeigen, und ein Abstand von größer als 0.412 bedeutet, dass die Bilder unterschiedliche Katzen zeigen. Auf unserer Testmenge mit ca. 2000 Bildern von ca. 600 verschiedenen Katzen erreichte die KI dann eine Klassifikationsgenauigkeit von ca. 96%. 
<br>
Das von uns erzielte Anwendungsszenario umfasst aber eigentlich die Sortierung von Vermisstenmeldungen nach Ähnlichkeit zu einem Bild, dass ein Finder einer potentiell vermissten Katze hochlädt. So können Katzen einfach per Handy-App, unabhängig von Kennzeichnungen wie Chips oder Tätowierungen, in Sekundenschnelle identifiziert werden. Dieses Szenario simulierten wir, indem wir die Katzen des Testdatensatzes nach Ähnlichkeit zu einem Referenzbild sortierten. Dabei befand sich in einer Liste mit 100 „Vermisstenmeldungen“ die gesuchte Katze in 97% der Fälle auf Platz 10 oder früher, und in 65% der Fälle auf Platz 1. Im Durchschnitt landete sie sogar auf Platz 2.6; der Median der Platzierungen lag bei nur 1.5. 

