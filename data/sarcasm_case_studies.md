# Qualitative Sarcasm Case Studies (NLP Evaluation Framework)

This dataset contains top positive interjections (> 0.95 sentiment score) called by 
Opposition factions towards Government speakers in the German Bundestag. 
Each record includes textual context (5 sentences before and 3 sentences after) for manual annotation.

### 1. Interjection #352230: „Zurufe von der AfD: Danke schön!“
- **Sentiment Score:** `0.9880`
- **Interjection Faction (Opposition):** AfD
- **Main Speaker (Government):** Marian Wendt (CDU/CSU)
- **Session:** Wahlperiode 19, Sitzung 13 (Date: 2018-02-21)
- **Context Before (5 sentences):** „*Den verletzten Beamten wünsche ich von hier aus gute Genesung. (Beifall bei der CDU/CSU sowie bei Abgeordneten der SPD, der FDP und des BÜNDNISSES 90/DIE GRÜNEN) Die Polizei hat nach dem Verhältnismäßigkeitsprinzip gehandelt. Es waren mehr als 1 500 Gegendemonstranten vor Ort. Es wurde mit der Räumung angefangen, Festnahmen und Identitätsfeststellungen erfolgten; denn jeder hat das Recht – auch Sie als AfD – zu demonstrieren. (*“
- **Context After (3 sentences):** „*) Und der Staat soll Ihnen das ermöglichen. Die Anmelderin selber hat die Demonstration beendet, obwohl die Räumungen bereits vonstattengingen. (Karsten Hilse [AfD]: Das stimmt nicht!) Wieder einmal haben hier aus meiner Sicht linksextreme Randalierer der Demokratie einen Bärendienst erwiesen und der AfD leider eine Steilvorlage für ihre Opferrolle geliefert.*“
- **Manual Annotation:**
  - `manual_label`: _[Sarcasm / Sincere_Positive / Ambiguous]_
  - `error_category`: _[Ironical_Agreement / Rhetorical_Politeness / Model_Misclassification]_
  - `notes`: _[]_

### 2. Interjection #360693: „Karsten Hilse [AfD]: Danke! Danke schön! – Weiterer Zuruf von der AfD: Yes we can!“
- **Sentiment Score:** `0.9871`
- **Interjection Faction (Opposition):** AfD
- **Main Speaker (Government):** Carsten Träger (SPD)
- **Session:** Wahlperiode 19, Sitzung 24 (Date: 2018-03-23)
- **Context Before (5 sentences):** „*Gute Umweltpolitik denkt das Soziale mit. (Beifall bei der SPD sowie bei Abgeordneten der CDU/CSU) Kolleginnen und Kollegen, das zentrale Umweltprojekt dieser Legislaturperiode wird der Klimaschutz sein. Wir wollen den Klimawandel begrenzen: in den nächsten zwölf Monaten mit einem Gesetz. Es wird Zeit. Herr Hilse, bei Ihrer Rede wehte ein Hauch von Trump durch den Saal – (*“
- **Context After (3 sentences):** „*) nach dem Motto: Ich habe eine Meinung. Verwirr mich nicht mit Tatsachen! (Beifall bei der SPD, der LINKEN und dem BÜNDNIS 90/DIE GRÜNEN sowie bei Abgeordneten der CDU/CSU) Es wird höchste Zeit, dass alle Sektoren ihren Beitrag leisten, so wie wir es im Klimaschutzplan bereits vereinbart haben.*“
- **Manual Annotation:**
  - `manual_label`: _[Sarcasm / Sincere_Positive / Ambiguous]_
  - `error_category`: _[Ironical_Agreement / Rhetorical_Politeness / Model_Misclassification]_
  - `notes`: _[]_

### 3. Interjection #87254: „Uwe Schulz [AfD]: Das wäre das Beste! Gute Idee!“
- **Sentiment Score:** `0.9864`
- **Interjection Faction (Opposition):** AfD
- **Main Speaker (Government):** Ann-Veruschka Jurisch (FDP)
- **Session:** Wahlperiode 20, Sitzung 97 (Date: 2023-04-20)
- **Context Before (5 sentences):** „*D.!) damals verfolgt hat, ex post oder auch ex ante mittragen. Und man kann auch nicht, wenn man 16 Jahre als Bundeskanzlerin gedient hat, immer alles richtig machen. (Stephan Brandner [AfD]: Nee! Aber öfter mal was richtig machen!) Aber es sollte uns gelingen, die Lebensleistung von Frau Bundeskanzlerin außer Dienst Merkel zumindest insoweit anzuerkennen, dass wir jetzt nicht darüber reden, (Stephan Brandner [AfD]: Sie hat unser Land ruiniert!) ihr die Würdigungen nicht zusprechen zu wollen, die mit diesem Orden verbunden sind, oder ihr den Orden womöglich noch abzuerkennen. (*“
- **Context After (3 sentences):** „*) Ich möchte auf das Diktum meines Kollegen Fricke von der Dankbarkeit und vom Dankesagen zurückkommen. Das ist doch wirklich wichtig. (Beifall bei Abgeordneten der FDP) Frau Merkel hat seit 1991 Verantwortung in unserem Land übernommen: als Ministerin, als MdB, als Parteivorsitzende, als Kanzlerin.*“
- **Manual Annotation:**
  - `manual_label`: _[Sarcasm / Sincere_Positive / Ambiguous]_
  - `error_category`: _[Ironical_Agreement / Rhetorical_Politeness / Model_Misclassification]_
  - `notes`: _[]_

### 4. Interjection #105709: „Kay Gottschalk [AfD]: Sie kennen meine Rede! Danke! Schöne Vorlage!“
- **Sentiment Score:** `0.9860`
- **Interjection Faction (Opposition):** AfD
- **Main Speaker (Government):** Melanie Wegling (SPD)
- **Session:** Wahlperiode 20, Sitzung 115 (Date: 2023-07-06)
- **Context Before (5 sentences):** „*Das ist ein konkreter, ein wichtiger Schritt, um die Attraktivität unseres Wirtschaftsstandorts zu verbessern. (Beifall bei der SPD sowie bei Abgeordneten des BÜNDNISSES 90/DIE GRÜNEN und der FDP) Hat die Union zugestimmt? Wohl kaum. Beim Stichwort „Fachkräftemangel“ mache ich gerne weiter. (*“
- **Context After (3 sentences):** „*) Außer mit mehr ausländischen Fachkräften müssen wir uns unbedingt mit einer weiteren, diesmal inländischen Reserve beschäftigen, und zwar mit den zahlreichen Frauen, die gerne mehr arbeiten würden, aber aus strukturellen Gründen darauf verzichten müssen. Mehrmals habe ich schon auf diese Problematik hingewiesen; ich mache es an dieser Stelle auch gerne noch mal. Nicht nur das DIW und die Wirtschaftsweisen sind sich einig: Stärkere Arbeitsanreize für die vielen nicht erwerbstätigen Frauen sind unverzichtbar, um dem Fachkräftemangel entgegenzuwirken und um zusätzliche Wachstumsimpulse zu generieren.*“
- **Manual Annotation:**
  - `manual_label`: _[Sarcasm / Sincere_Positive / Ambiguous]_
  - `error_category`: _[Ironical_Agreement / Rhetorical_Politeness / Model_Misclassification]_
  - `notes`: _[]_

### 5. Interjection #40360: „Artur Auernhammer [CDU/CSU]: Danke schön! – Marianne Schieder [SPD]: Vor Bayern den Hut zu ziehen, ist immer gut!“
- **Sentiment Score:** `0.9846`
- **Interjection Faction (Opposition):** CDU/CSU
- **Main Speaker (Government):** Isabel Mackensen-Geis (SPD)
- **Session:** Wahlperiode 20, Sitzung 51 (Date: 2022-09-08)
- **Context Before (5 sentences):** „*(Beifall bei Abgeordneten der SPD und des BÜNDNISSES 90/DIE GRÜNEN) Wenn wir unsere Wälder für den Klimawandel fitmachen wollen, ist die nachhaltige Anpassung der Rehwildbestände erforderlich; denn Schutzmaßnahmen wie Zäune und Verbisshüllen kosten richtig viel Geld. Bei den Zäunen rechnet man mit etwa 15 Euro pro laufenden Meter; da kommt man bei einem Hektar schon auf zusätzliche Kosten von 6 000 Euro. Beim Einzelbaumschutz kann man bei etwa 10 Euro pro Pflanze – Entsorgungskosten noch nicht mit eingerechnet – bei einer Neupflanzung von einem Hektar Waldfläche mit rund 2 500 jungen Bäumen schnell mit zusätzlichen Kosten von 25 000 Euro rechnen. In diesem Falle ziehe ich ausnahmsweise – ich gebe es zu – meinen Hut vor der bayerischen Gesetzgebung. (*“
- **Context After (3 sentences):** „*) Hier ist die Zielsetzung – ich zitiere –, „einen standortgemäßen und möglichst naturnahen Zustand des Waldes unter Berücksichtigung des Grundsatzes ‚Wald vor Wildʼ zu bewahren oder herzustellen“, im Bayerischen Waldgesetz festgeschrieben. (Beifall bei der SPD sowie bei Abgeordneten des BÜNDNISSES 90/DIE GRÜNEN und der FDP – Artur Auernhammer [CDU/CSU]: Von Bayern lernen, heißt siegen lernen!) Für die AfD-Fraktion hat das Wort Ulrike Schielke-Ziesing. (Beifall bei der AfD)*“
- **Manual Annotation:**
  - `manual_label`: _[Sarcasm / Sincere_Positive / Ambiguous]_
  - `error_category`: _[Ironical_Agreement / Rhetorical_Politeness / Model_Misclassification]_
  - `notes`: _[]_

### 6. Interjection #90281: „Enrico Komning [AfD]: Das wäre schön! Ein guter Ansatz!“
- **Sentiment Score:** `0.9839`
- **Interjection Faction (Opposition):** AfD
- **Main Speaker (Government):** Rainer Semet (FDP)
- **Session:** Wahlperiode 20, Sitzung 100 (Date: 2023-04-27)
- **Context Before (5 sentences):** „*Begleiten wir dieses Land mit seinen 120 Millionen Einwohnern in eine bessere, friedlichere Zukunft. Begreifen wir die strategische Bedeutung von Frieden und Stabilität in Tigray für Äthiopien und für die gesamte Region um das Horn von Afrika. Ich persönlich halte – da bin ich nicht alleine in diesem Haus – eine nachhaltige wirtschaftliche Zusammenarbeit mit den Ländern Afrikas für eine ganz wichtige Aufgabe. (Beifall bei der FDP sowie bei Abgeordneten der SPD und des BÜNDNISSES 90/DIE GRÜNEN) Gewinnen wir das Vertrauen, indem wir ihnen zeigen, dass wir da sind, dass wir zuhören, dass wir präsent sind, dass wir nicht als Lehrmeister kommen, sondern als echte verlässliche Partner. (*“
- **Context After (3 sentences):** „*) Gewinnen wir das Vertrauen, indem wir vor allem auch regionale Organisationen und Akteure stärken. Sie wissen am besten, welche Maßnahmen der Zusammenarbeit am nötigsten sind und den größten Erfolg versprechen. Seien wir dadurch der attraktivere Partner im Vergleich zu Russland und China.*“
- **Manual Annotation:**
  - `manual_label`: _[Sarcasm / Sincere_Positive / Ambiguous]_
  - `error_category`: _[Ironical_Agreement / Rhetorical_Politeness / Model_Misclassification]_
  - `notes`: _[]_

### 7. Interjection #87749: „Stephan Brandner [AfD]: Sehr schön! Können Sie gleich hier abgeben! Ich reiche das weiter!“
- **Sentiment Score:** `0.9837`
- **Interjection Faction (Opposition):** AfD
- **Main Speaker (Government):** Hagen Reinhold (FDP)
- **Session:** Wahlperiode 20, Sitzung 97 (Date: 2023-04-20)
- **Context Before (5 sentences):** „*Das Recht der Selbstverteidigung gegen einen Angriffskrieg hat die UN-Vollversammlung schon im März 2022 bestätigt. Das alles ist nicht ganz einfach; das verstehe ich. Deshalb bin ich als Teil einer regierungstragenden Servicefraktion ganz vorbildlich und habe Ihnen ein Schaubild mitgebracht, das den Genehmigungsprozess beim Export von Kriegswaffen darstellt. Den können Sie sich dann noch mal angucken und verinnerlichen; dann wissen Sie, um was es geht. (*“
- **Context After (3 sentences):** „*) Weil das aber alles nicht so einfach ist und es manchem sogar schwerfällt, das Ganze zu durchdringen, ist die Bundesregierung dabei, ein Rüstungsexportkontrollgesetz zu erarbeiten. Das hätten Sie gut als Aufhänger für Ihren Antrag nutzen können; das haben Sie aber nicht gemacht. Ihr Antrag sieht da nichts vor.*“
- **Manual Annotation:**
  - `manual_label`: _[Sarcasm / Sincere_Positive / Ambiguous]_
  - `error_category`: _[Ironical_Agreement / Rhetorical_Politeness / Model_Misclassification]_
  - `notes`: _[]_

### 8. Interjection #91647: „Sebastian Brehm [CDU/CSU]: Die ist gut! Die ist hervorragend!“
- **Sentiment Score:** `0.9819`
- **Interjection Faction (Opposition):** CDU/CSU
- **Main Speaker (Government):** Maximilian Mordhorst (FDP)
- **Session:** Wahlperiode 20, Sitzung 101 (Date: 2023-04-28)
- **Context Before (5 sentences):** „*Es hätte mich nur gefreut, wenn Sie das auch mal in einer Bundesregierung getan hätten. (Dr. Klaus Wiener [CDU/CSU]: Haben wir doch! Vollbeschäftigung!) Denn was haben wir vorgefunden, als wir angefangen haben, zu regieren? Ich erinnere an das ein oder andere: nicht nachhaltig finanzierte Sozialsysteme, Rente mit 63, Mütterrente – (*“
- **Context After (3 sentences):** „*) damit haben Sie grundsätzliche Prinzipien der Rentenversicherung gebrochen –, nicht nachhaltig finanzierte Pflegeversicherung. Sie haben mitgetragen, dass bei den Coronawirtschaftshilfen die Schleusentore der Schuldenbremse geöffnet wurden. Stellen Sie sich doch einmal selbst die kritische Frage: (Zuruf des Abg.*“
- **Manual Annotation:**
  - `manual_label`: _[Sarcasm / Sincere_Positive / Ambiguous]_
  - `error_category`: _[Ironical_Agreement / Rhetorical_Politeness / Model_Misclassification]_
  - `notes`: _[]_

### 9. Interjection #56780: „Patrick Schnieder [CDU/CSU]: Danke! Super! Schön, dass Sie das sagen!“
- **Sentiment Score:** `0.9817`
- **Interjection Faction (Opposition):** CDU/CSU
- **Main Speaker (Government):** Laura Kraft (BÜNDNIS 90/DIE GRÜNEN)
- **Session:** Wahlperiode 20, Sitzung 67 (Date: 2022-11-11)
- **Context Before (5 sentences):** „*Somit müssen wir noch mal über Ressortzuständigkeiten sprechen; denn das Portal www.energieforschung.de ist in der Zuständigkeit des BMWK, nicht des BMBF, und die Seite gibt es schon seit 2018, also auch schon unter Ihrer Regierung. Und wenn Sie auf der Seite des BMBF unterwegs sind, dann finden Sie einen Bericht zum 7. Energieforschungsprogramm von 2020, also auch aus Ihrer Regierungszeit. Es geht also gar nicht darum, herauszustellen, was nicht gemacht wurde, sondern es ist ja schon viel gefördert worden, ja, auch unter der Vorgängerregierung, also unter Ihrer Regierung. (*“
- **Context After (3 sentences):** „*) Es ist somit nicht so, als hätten Sie dieses Thema jetzt noch mal ganz neu aufgebracht und das ganz neu entdeckt, sondern da ist ja schon länger viel passiert. (Zuruf des Abg. Thomas Jarzombek [CDU/CSU]) Und da frage ich mich: Was ist denn Ihr Motto?*“
- **Manual Annotation:**
  - `manual_label`: _[Sarcasm / Sincere_Positive / Ambiguous]_
  - `error_category`: _[Ironical_Agreement / Rhetorical_Politeness / Model_Misclassification]_
  - `notes`: _[]_

### 10. Interjection #217341: „Dr. Valerie Wilms [BÜNDNIS 90/DIE GRÜNEN]: Danke! Sehr schön!“
- **Sentiment Score:** `0.9813`
- **Interjection Faction (Opposition):** BÜNDNIS 90/DIE GRÜNEN
- **Main Speaker (Government):** Andreas Lenz (CDU/CSU)
- **Session:** Wahlperiode 18, Sitzung 17 (Date: None)
- **Context Before (5 sentences):** „*– Wer enthält sich? – Der Antrag ist einstimmig angenommen. Ich wünsche von Herzen eine gute Arbeit in diesem wirklich wichtigen Gremium. Damit ist der Parlamentarische Beirat für nachhaltige Entwicklung eingesetzt. (*“
- **Context After (3 sentences):** „*) Ich rufe den Tagesordnungspunkt 9 auf:*“
- **Manual Annotation:**
  - `manual_label`: _[Sarcasm / Sincere_Positive / Ambiguous]_
  - `error_category`: _[Ironical_Agreement / Rhetorical_Politeness / Model_Misclassification]_
  - `notes`: _[]_

### 11. Interjection #306113: „Kai Gehring [BÜNDNIS 90/DIE GRÜNEN]: Ein sehr positives!“
- **Sentiment Score:** `0.9811`
- **Interjection Faction (Opposition):** BÜNDNIS 90/DIE GRÜNEN
- **Main Speaker (Government):** Markus Koob (CDU/CSU)
- **Session:** Wahlperiode 18, Sitzung 179 (Date: 2016-06-23)
- **Context Before (5 sentences):** „*Das ist zuallererst ein Erfolg der Community und ihrer vielen Unterstützerinnen und Unterstützer. Informationsmaterialien, wie Sie sie in Ihrem Antrag fordern, gibt es meines Erachtens von den Vereinen und Verbänden der Community schon heute in ausreichender Zahl. Manchmal erreicht man durch Quantität eher das Gegenteil von dem, was man sich wünscht. Qualitativ hochwertige Informationen findet man bereits heute zahlreich im Internet, ebenso Beratungsangebote, auch zielgruppenspezifische. Zudem frage ich mich, welches Bild Sie von der Kinder- und Jugendhilfe in Deutschland haben, (*“
- **Context After (3 sentences):** „*) wenn Sie es für nötig erachten, das Sozialgesetzbuch VIII zu ergänzen, um die Grundrichtung der Erziehung der LSBTI sicherzustellen. (Kai Gehring [BÜNDNIS 90/DIE GRÜNEN]: Sie sind doch so gut in die Rede gestartet!) Ihrem Wunsch nach einer alle vier Jahre erscheinenden Studie zur Lebenssituation lesbischer, schwuler, bisexueller, trans- und intergeschlechtlicher Jugendlicher kann ich durchaus etwas abgewinnen. Der Rest Ihres Antrags besteht aus Forderungen nach Aktionsplänen, Pilotprojekten, Kampagnen und Koordinierungsstellen – viele neue Bürokratiestrukturen, anstatt existierende Strukturen einfach zu nutzen und weiterzuentwickeln.*“
- **Manual Annotation:**
  - `manual_label`: _[Sarcasm / Sincere_Positive / Ambiguous]_
  - `error_category`: _[Ironical_Agreement / Rhetorical_Politeness / Model_Misclassification]_
  - `notes`: _[]_

### 12. Interjection #473036: „Matthias W. Birkwald [DIE LINKE]: Ja, gute Idee! Das wäre gut!“
- **Sentiment Score:** `0.9801`
- **Interjection Faction (Opposition):** DIE LINKE
- **Main Speaker (Government):** Max Straubinger (CDU/CSU)
- **Session:** Wahlperiode 19, Sitzung 158 (Date: 2020-05-07)
- **Context Before (5 sentences):** „*Frau Hiller-Ohm hat ja bereits herausgestellt, dass wir die Pensionskassen in den besonderen Schutz des Pensions-Sicherungs-Vereins aufnehmen. Das ist ein wichtiger Punkt. Hier gab es sicherlich auch unterschiedliche Sichtweisen. Es gab die Forderung, auch die Firmendirektversicherungen direkt unter diesen Schutz zu stellen. (*“
- **Context After (3 sentences):** „*) Sie sind aber im Rahmen von Protektor sowieso geschützt. (Matthias W. Birkwald [DIE LINKE]: Protektor ist deutlich schlechter als der Pensions-Sicherungs-Verein!) – Ich weiß, Herr Birkwald, Sie setzen auf nichtprivatwirtschaftliche Lösungen.*“
- **Manual Annotation:**
  - `manual_label`: _[Sarcasm / Sincere_Positive / Ambiguous]_
  - `error_category`: _[Ironical_Agreement / Rhetorical_Politeness / Model_Misclassification]_
  - `notes`: _[]_

### 13. Interjection #193435: „Stephan Brandner [AfD]: Danke schön!“
- **Sentiment Score:** `0.9800`
- **Interjection Faction (Opposition):** AfD
- **Main Speaker (Government):** Tobias Bacherle (BÜNDNIS 90/DIE GRÜNEN)
- **Session:** Wahlperiode 20, Sitzung 195 (Date: 2024-10-18)
- **Context Before (5 sentences):** „*Sehr geehrter Herr Präsident! Liebe Kolleginnen und Kollegen! Herr Brandner hat gerade den letzten „Achtung, Reichelt!“-Text hier vorgelesen – nicht besonders gelungen, (Lachen des Abg. Stephan Brandner [AfD]) aber immerhin gut rezipiert. (*“
- **Context After (3 sentences):** „*) Da klingelt einem so ein bisschen Horst Seehofers „Die Grünen sind an allem schuld“ im Ohr. (Stephan Brandner [AfD]: Horst Seehofer?) Wir sind das vielleicht schon gewohnt, aber inhaltlich besonders fundiert wird es dadurch halt leider nicht. Ich habe mir überlegt – (Stephan Brandner [AfD]: Betreutes Denken!) die Kollegin hatte das gerade gesagt: wir haben das im Januar und im März hier besprochen –: Wie kommt es denn, dass Sie jetzt so late to the party sind?*“
- **Manual Annotation:**
  - `manual_label`: _[Sarcasm / Sincere_Positive / Ambiguous]_
  - `error_category`: _[Ironical_Agreement / Rhetorical_Politeness / Model_Misclassification]_
  - `notes`: _[]_

### 14. Interjection #560140: „Stephan Brandner [AfD]: Danke schön!“
- **Sentiment Score:** `0.9800`
- **Interjection Faction (Opposition):** AfD
- **Main Speaker (Government):** Helge Lindh (SPD)
- **Session:** Wahlperiode 21, Sitzung 22 (Date: 2025-09-12)
- **Context Before (5 sentences):** „*– Dr. Malte Kaufmann [AfD]: Vor allem das blaue Hemd ist sehr schön!) ist kein Geheimauftrag von Baerbock und Habeck. Was wir heute erleben, ist ein großer Zaubertrick der AfD – und das kann ich auch lückenlos nachweisen –, (Stephan Brandner [AfD]: Abrakadabra! – Tobias Ebenberger [AfD]: Abrakadabra!) ein wirklich perfides, auf die Spitze getriebenes Täuschungs- und Vexierspiel und letztlich eine leider perfektionierte Täter-Opfer-Umkehr, die Sie betreiben. (*“
- **Context After (3 sentences):** „*) Wovon reden wir nun im Zusammenhang mit den Fällen von § 188 StGB? Leute schreiben mir zum Beispiel – ich zitiere –: „Helge, der gesichert linksextreme Schwule“, „Das ist eine typische Pädophilenfresse“, „Es tut mir wirklich leid, wie man so furchtbar von der Natur bestraft werden kann“, „Das kommt dabei raus, wenn Bruder und Schwester Liebe machen“ (Dr. Malte Kaufmann [AfD]: Aber das ist ja auch sonst strafbar!*“
- **Manual Annotation:**
  - `manual_label`: _[Sarcasm / Sincere_Positive / Ambiguous]_
  - `error_category`: _[Ironical_Agreement / Rhetorical_Politeness / Model_Misclassification]_
  - `notes`: _[]_

### 15. Interjection #634912: „Stephan Brandner [AfD]: Danke schön!“
- **Sentiment Score:** `0.9800`
- **Interjection Faction (Opposition):** AfD
- **Main Speaker (Government):** Detlef Seif (CDU/CSU)
- **Session:** Wahlperiode 21, Sitzung 97 (Date: 2026-09-25)
- **Context Before (5 sentences):** „*Der wirksame Schutz unserer Grenzen ist ein wichtiges Thema. Gerade weil im Schengenraum Freizügigkeit herrscht und damit die Errichtung von Grenzbefestigungen ausgeschlossen ist, müssen wir die vorhandenen Möglichkeiten nutzen, um Organisierte Kriminalität, insbesondere die Schleppertätigkeit und auch die irreguläre Migration, zu bekämpfen. (Beifall bei der CDU/CSU sowie des Abg. Ingo Vogel [SPD]) Der Antrag ist insoweit nicht falsch. (*“
- **Context After (3 sentences):** „*) Aber er geht in die falsche Richtung, (Stephan Brandner [AfD]: Nicht „danke schön“!) weil er die Möglichkeiten, die die Bundespolizei schon heute hat, nicht berücksichtigt. Die Bundespolizei arbeitet bereits erfolgreich mit der Integrierten Grenzkontrollanwendung, kurz: IGA. Diese verknüpft Abfragen, Fahndungsdaten und Dokumentenprüfung in einem System.*“
- **Manual Annotation:**
  - `manual_label`: _[Sarcasm / Sincere_Positive / Ambiguous]_
  - `error_category`: _[Ironical_Agreement / Rhetorical_Politeness / Model_Misclassification]_
  - `notes`: _[]_

### 16. Interjection #58760: „Stephan Brandner [AfD]: „Liebe AfD“! Das ist ja nett von Ihnen! Danke schön!“
- **Sentiment Score:** `0.9798`
- **Interjection Faction (Opposition):** AfD
- **Main Speaker (Government):** Achim Post (SPD)
- **Session:** Wahlperiode 20, Sitzung 69 (Date: 2022-11-23)
- **Context Before (5 sentences):** „*Dass Sie eine andere Sichtweise als die Bundesregierung haben, zeichnet solche Debatten in diesem Parlament aus. Das alles kann man – da gucke ich nach ganz rechts außen – nicht von allen Rednerinnen und Rednern sagen. Da ist unter Formulierungen wie „der Niedergang Deutschlands“, „das Ende des europäischen Abendlandes“, „der Niedergang der deutschen Kultur“ und generell „der Untergang Deutschlands“ nichts zu machen. Liebe AfD, dass Sie von Demokratie nicht viel halten, das weiß ich schon. (*“
- **Context After (3 sentences):** „*) – Ja, liebe AfD, ich bin nett. – Ich weiß, dass Sie von Demokratie nicht viel halten. Von Deutschland, von diesem Land, von unserem Land verstehen Sie gar nichts, liebe Kolleginnen und Kollegen.*“
- **Manual Annotation:**
  - `manual_label`: _[Sarcasm / Sincere_Positive / Ambiguous]_
  - `error_category`: _[Ironical_Agreement / Rhetorical_Politeness / Model_Misclassification]_
  - `notes`: _[]_

### 17. Interjection #577110: „Martin Erwin Renner [AfD]: Das ist lieb! Danke!“
- **Sentiment Score:** `0.9795`
- **Interjection Faction (Opposition):** AfD
- **Main Speaker (Government):** Holger Mann (SPD)
- **Session:** Wahlperiode 21, Sitzung 39 (Date: 2025-11-12)
- **Context Before (5 sentences):** „*Götz Frömming [AfD]: Nee, darum geht es nicht! Nein, das ist falsch!) Sie drehen es sich also, wie Sie es brauchen. (Beifall bei Abgeordneten der SPD und der CDU/CSU) Herr Abgeordneter, gestatten Sie eine Zwischenfrage aus der AfD-Fraktion? Bitte schön. (*“
- **Context After (3 sentences):** „*) Vielen Dank, Herr Kollege Mann, dass Sie die Zwischenfrage zulassen. Ich möchte nur mal darauf hinweisen: Es geht nicht um den völlig normalen Vorgang, dass aus Reden zitiert wird, egal ob aus unserer oder Ihrer Fraktion. Es geht darum, dass hier komplette Reden, komplette Artikel auf der Seite veröffentlicht worden sind (Pascal Reddig [CDU/CSU]: Das ist auch zulässig!) – das ist auch zulässig, richtig – und der Anschein erweckt wurde, es würde sich hierbei um Autoren ebendieses Magazins handeln.*“
- **Manual Annotation:**
  - `manual_label`: _[Sarcasm / Sincere_Positive / Ambiguous]_
  - `error_category`: _[Ironical_Agreement / Rhetorical_Politeness / Model_Misclassification]_
  - `notes`: _[]_

### 18. Interjection #590140: „Stephan Brandner [AfD]: Das ist auch ein AfD-Thema! Danke schön!“
- **Sentiment Score:** `0.9794`
- **Interjection Faction (Opposition):** AfD
- **Main Speaker (Government):** Thomas Jarzombek (CDU/CSU)
- **Session:** Wahlperiode 21, Sitzung 51 (Date: 2025-12-19)
- **Context Before (5 sentences):** „*Demnach sollen dadurch der Datenschutz der US-Nutzer, die Sicherheit des Algorithmus, die Moderation der Inhalte und die Softwaresicherheit verantwortet werden. Der Empfehlungsalgorithmus soll auf Basis der US-Nutzerdaten neu trainiert werden, um sicherzustellen, dass keine externe Manipulation stattfindet. Ich glaube, dass all das unsere Debatte hier bereichert. Die Frage ist ja, ob der DSA eigentlich sein Ziel erreicht hat, nämlich ein sicheres und freies Internet, in dem Meinungspluralismus besteht und in dem wir nicht von Externen indoktriniert werden. Deshalb, meine Damen und Herren, ist das doch ein gutes Thema für die letzte Debatte in diesem Jahr (*“
- **Context After (3 sentences):** „*) und eine gute Aufgabe für das nächste Jahr, darüber nachzudenken, wie wir gemeinsam dieses freie und sichere Internet gewährleisten können. Vielen Dank. (Beifall bei der CDU/CSU sowie bei Abgeordneten der SPD und der Abg.*“
- **Manual Annotation:**
  - `manual_label`: _[Sarcasm / Sincere_Positive / Ambiguous]_
  - `error_category`: _[Ironical_Agreement / Rhetorical_Politeness / Model_Misclassification]_
  - `notes`: _[]_

### 19. Interjection #497856: „Zuruf von der FDP: Sehr schön!“
- **Sentiment Score:** `0.9794`
- **Interjection Faction (Opposition):** FDP
- **Main Speaker (Government):** Tobias Zech (CDU/CSU)
- **Session:** Wahlperiode 19, Sitzung 186 (Date: 2020-10-29)
- **Context Before (5 sentences):** „*Das ist notwendig; denn wir haben es mit unterschiedlichen Lebensläufen und unterschiedlichen Berufsentwicklungen zu tun. Ich selbst arbeite jetzt seit 23 Jahren. Ich habe sowohl eine private Vorsorge als auch eine betriebliche Altersvorsorge. Ich habe auch in die gesetzliche Rentenversicherung einbezahlt und bekomme auch hier eine Vorsorge. (*“
- **Context After (3 sentences):** „*) Und was mache ich jedes Jahr? Ich nehme alle Statusmitteilungen, loche sie und hefte sie ab. Ziel ist es, dass wir die Anwartschaften, die Statusmitteilungen und die Vorsorgeeinrichtungen, die Standmitteilungen erbringen müssen, zusammenfassen.*“
- **Manual Annotation:**
  - `manual_label`: _[Sarcasm / Sincere_Positive / Ambiguous]_
  - `error_category`: _[Ironical_Agreement / Rhetorical_Politeness / Model_Misclassification]_
  - `notes`: _[]_

### 20. Interjection #223183: „Klaus Ernst [DIE LINKE]: Danke schön! Dann kann ich mich ja wieder setzen!“
- **Sentiment Score:** `0.9793`
- **Interjection Faction (Opposition):** DIE LINKE
- **Main Speaker (Government):** Andreas G. Lämmel (CDU/CSU)
- **Session:** Wahlperiode 18, Sitzung 30 (Date: 2014-04-10)
- **Context Before (5 sentences):** „*Dazu habe ich bis jetzt noch nichts von Ihnen gehört. Aber das ist auch ein Problem der langen Redezeiten, die den Regierungsfraktionen zurzeit zur Verfügung stehen: Da kann man auch ein bisschen blödeln. Ich werde Ihre Frage im Laufe meiner Rede beantworten. Sie können damit rechnen, dass Sie eine Antwort bekommen. (*“
- **Context After (3 sentences):** „*) „Stärkung der Wettbewerbsfähigkeit Deutschlands“: Unter diesem Motto standen die Koalitionsverhandlungen. Genau darum geht es auch bei diesem Haushalt: Wir müssen die Position der deutschen Volkswirtschaft in der Welt stärken. Denn die Amerikaner, Asiaten und Araber, sie alle schlafen nicht und sind drauf und dran, sich Anteile am wirtschaftlichen Geschehen der Welt zu erobern.*“
- **Manual Annotation:**
  - `manual_label`: _[Sarcasm / Sincere_Positive / Ambiguous]_
  - `error_category`: _[Ironical_Agreement / Rhetorical_Politeness / Model_Misclassification]_
  - `notes`: _[]_

### 21. Interjection #312452: „Zuruf von der LINKEN: Die war gut!“
- **Sentiment Score:** `0.9792`
- **Interjection Faction (Opposition):** DIE LINKE
- **Main Speaker (Government):** Markus Paschke (SPD)
- **Session:** Wahlperiode 18, Sitzung 190 (Date: 2016-09-22)
- **Context Before (5 sentences):** „*Sehr geehrte Frau Präsidentin! Liebe Kolleginnen und Kollegen! Meine sehr geehrten Damen und Herren! Bevor ich etwas zum Gesetz sage, muss ich etwas zu Ihrer Rede sagen, Frau Wagenknecht. (*“
- **Context After (3 sentences):** „*) Ich fand, die war unterirdisch. (Beifall bei der SPD und der CDU/CSU – Zuruf von der LINKEN: Ach Quatsch, Markus!) Das war ein Populismuswettbewerb mit der anderen Seite, die hier nichts zu suchen hat. Ich muss ehrlich sagen: So stellt die Linke auch keine Alternative dar.*“
- **Manual Annotation:**
  - `manual_label`: _[Sarcasm / Sincere_Positive / Ambiguous]_
  - `error_category`: _[Ironical_Agreement / Rhetorical_Politeness / Model_Misclassification]_
  - `notes`: _[]_

### 22. Interjection #608481: „Tino Chrupalla [AfD]: „Einstellen“, sehr schön!“
- **Sentiment Score:** `0.9786`
- **Interjection Faction (Opposition):** AfD
- **Main Speaker (Government):** Mechthilde Wittmann (CDU/CSU)
- **Session:** Wahlperiode 21, Sitzung 71 (Date: 2026-04-16)
- **Context Before (5 sentences):** „*Denn die Erfahrungen sind etwas differenzierter, als Sie es gerade dargestellt haben, und es ist auch nachgewiesen. Wir können in der Rückschau das Ganze nachrechnen. Das ifo-Institut hat schon im Juni 2022 genau nachgewiesen, dass beim Diesel 100 Prozent der Steuersenkungen im ersten Monat weitergegeben wurden und im zweiten Monat noch mal weitere 85 Prozent. (Widerspruch der Abg. Katharina Beck [BÜNDNIS 90/DIE GRÜNEN]) Das heißt: Ja, so ganz komplett wurde er nicht weitergegeben, aber so weitgehend, dass wir davon sprechen können, dass Tankrabatte tatsächlich eine Wirkung haben, solange sie befristet sind und nur eine Überbrückung darstellen, bis die Menschen sich dann tatsächlich darauf einstellen (*“
- **Context After (3 sentences):** „*) und die Maßnahmen greifen konnten. Lassen Sie mich zum Kartellrecht kommen, das ich ohnehin noch angesprochen hätte. Das Kartellrecht haben wir jetzt nachgeschärft, indem wir gesagt haben: Es gilt eine Beweislastumkehr.*“
- **Manual Annotation:**
  - `manual_label`: _[Sarcasm / Sincere_Positive / Ambiguous]_
  - `error_category`: _[Ironical_Agreement / Rhetorical_Politeness / Model_Misclassification]_
  - `notes`: _[]_

### 23. Interjection #532840: „Jürgen Braun [AfD]: Das machen Sie super! Das machen Sie stark!“
- **Sentiment Score:** `0.9785`
- **Interjection Faction (Opposition):** AfD
- **Main Speaker (Government):** Martin Rabanus (SPD)
- **Session:** Wahlperiode 19, Sitzung 228 (Date: 2021-05-07)
- **Context Before (5 sentences):** „*– Zuruf der Abg. Helin Evrim Sommer [DIE LINKE]) Ich habe das öffentlich bisher nicht kommentiert. Da bin ich bei dem Kollegen der FDP: Man kann das alles mit ein bisschen weniger Aufgeregtheit machen. Das ist ein bisschen die Abteilung: Jeder blamiert sich so gut, wie er kann. (*“
- **Context After (3 sentences):** „*) Aber wenn diejenigen, die das machen, so machen, dann bekommen sie das Wort deswegen ja nicht abgeschnitten, und sie bekamen das Wort auch nicht abgeschnitten. Aber wer sich bewusst oder unbewusst einer gefährlichen rechtspopulistischen Erzählung annähert (Zurufe von der AfD: Oh!) und sich diese zunutze macht und sich da auch vor den Karren spannen lässt, (Zuruf von der AfD: Widerliche Hetze!) der muss dann auch mit der Meinungsfreiheit der anderen leben (Martin Hess [AfD]: Cancel Culture ist keine Meinungsfreiheit!) und ertragen, dass man das für gefährlichen Blödsinn hält. Das ist auch bei mir der Fall.*“
- **Manual Annotation:**
  - `manual_label`: _[Sarcasm / Sincere_Positive / Ambiguous]_
  - `error_category`: _[Ironical_Agreement / Rhetorical_Politeness / Model_Misclassification]_
  - `notes`: _[]_

### 24. Interjection #443678: „Zuruf von der AfD: Danke! Großzügig!“
- **Sentiment Score:** `0.9783`
- **Interjection Faction (Opposition):** AfD
- **Main Speaker (Government):** Martin Rabanus (SPD)
- **Session:** Wahlperiode 19, Sitzung 123 (Date: 2019-11-06)
- **Context Before (5 sentences):** „*Kein Problem!) Aber um das auch gleich von vornherein zu sagen: Das ist Ihr gutes parlamentarisches Recht. Das will Ihnen hier auch niemand abschneiden. Sie können das machen. Sie können hier im Rahmen der parlamentarischen Ordnung auch alles sagen, was immer Sie mögen – egal wie unsinnig das sein mag. (*“
- **Context After (3 sentences):** „*) Sie sind auch in den Medien präsent; Sie sind in Talkshows – manchmal inflationär – präsent. (Dr. Alice Weidel [AfD]: Ja, 3,3 Prozent bei den Öffentlich-Rechtlichen!*“
- **Manual Annotation:**
  - `manual_label`: _[Sarcasm / Sincere_Positive / Ambiguous]_
  - `error_category`: _[Ironical_Agreement / Rhetorical_Politeness / Model_Misclassification]_
  - `notes`: _[]_

### 25. Interjection #16661: „Matthias W. Birkwald [DIE LINKE]: Guter Song, Otto! Der Song war gut!“
- **Sentiment Score:** `0.9782`
- **Interjection Faction (Opposition):** DIE LINKE
- **Main Speaker (Government):** Otto Fricke (FDP)
- **Session:** Wahlperiode 20, Sitzung 26 (Date: 2022-03-25)
- **Context Before (5 sentences):** „*Dabei fallen mir zwei Dinge auf. Diese Koalition hat es innerhalb des halben Jahres geschafft, sich schlagkräftig zu formieren, beim Haushalt dafür zu sorgen, die Realitäten anzuerkennen, und gleichzeitig auf das, was an neuen Dingen auf uns zukommt, zu reagieren. Das ist etwas, was man dieser Koalition nicht zugetraut hat; aber ich bin stolz darauf, dass wir das bis hierhin geschafft haben, auch wenn jetzt die Arbeit im Ausschuss beginnt. (Beifall bei der FDP, der SPD und dem BÜNDNIS 90/DIE GRÜNEN) Wenn ich auf die eigentlich einzig ernstzunehmende Opposition und gleichzeitig auf dieses schöne Sonnenwetter draußen schaue, fällt mir nur Herwig Mitteregger mit dem Liedtext „Immer mehr, immer mehr, immer mehr“ von 1985 ein. (*“
- **Context After (3 sentences):** „*) Wenn ich nämlich sehe, was hier passiert ist, dann muss ich wirklich sagen: Wir wollen im parlamentarischen Diskurs doch die Diskussion mit Ihnen über Ihre Inhalte, über Ihre Ideen, um unsere – wie wir meinen – besseren Ideen mit Ihren zu messen und dann zu einer Entscheidung zu kommen. Dazu müsste von Ihnen etwas kommen. Daher habe ich mein Büro mal gebeten, in jeder Rede von Ihnen zu gucken, wo von Ihnen mehr gefordert wird.*“
- **Manual Annotation:**
  - `manual_label`: _[Sarcasm / Sincere_Positive / Ambiguous]_
  - `error_category`: _[Ironical_Agreement / Rhetorical_Politeness / Model_Misclassification]_
  - `notes`: _[]_

### 26. Interjection #561291: „Stephan Protschka [AfD]: Ja, super! So sauber und kleinlich arbeiten wir! Perfekt!“
- **Sentiment Score:** `0.9782`
- **Interjection Faction (Opposition):** AfD
- **Main Speaker (Government):** Oliver Vogt (CDU/CSU)
- **Session:** Wahlperiode 21, Sitzung 23 (Date: 2025-09-16)
- **Context Before (5 sentences):** „*Die AfD hat dieses Sprichwort aber anscheinend etwas zu wörtlich genommen. Sie haben tatsächlich eine Reihe von Änderungsanträgen eingebracht, um einzelne Haushaltstitel um sage und schreibe 1 000 Euro abzusenken. Um Ihnen das noch einmal darzulegen: Bei einem Haushaltsvolumen von 500 Milliarden Euro kürzen Sie, um sozusagen Wirkung zu entfalten, 1 000 Euro. Das sind, um es als Bruchzahl auszudrücken, zwei Milliardstel oder, anders gesprochen, eine Null, dann ein Komma, danach kommen noch mal acht Nullen, und dann kommt die erste Zahl über null. (*“
- **Context After (3 sentences):** „*) Das ist sehr ambitioniert, meine Damen und Herren von der AfD, sehr ambitioniert. (Beifall bei der CDU/CSU sowie des Abg. Esra Limbacher [SPD]) Damit haben Sie allerdings – das will ich Ihnen zugestehen – sehr eindrucksvoll unter Beweis gestellt, wie konzept- und substanzlos Ihre Haushaltspolitik ist.*“
- **Manual Annotation:**
  - `manual_label`: _[Sarcasm / Sincere_Positive / Ambiguous]_
  - `error_category`: _[Ironical_Agreement / Rhetorical_Politeness / Model_Misclassification]_
  - `notes`: _[]_

### 27. Interjection #569814: „Zuruf von der Linken: Der war gut!“
- **Sentiment Score:** `0.9781`
- **Interjection Faction (Opposition):** DIE LINKE
- **Main Speaker (Government):** Fritz Güntzler (CDU/CSU)
- **Session:** Wahlperiode 21, Sitzung 31 (Date: 2025-10-09)
- **Context Before (5 sentences):** „*Wir wollen kein Freibier für alle – 40 Milliarden Euro – ohne jede Gegenfinanzierung. Was Sie hier fordern, ist unsolide. Herr Abgeordneter, langsam kommen Sie ins Sportziel. (Heiterkeit bei Abgeordneten der CDU/CSU und der SPD) Oh, da gab es eine Auswechslung. (*“
- **Context After (3 sentences):** „*) Herr Präsident, ich komme zum Ende: Vielen Dank. (Beifall bei der CDU/CSU sowie bei Abgeordneten der SPD) Für Bündnis 90/Die Grünen darf ich Tina Winklmann aufrufen. (Beifall beim BÜNDNIS 90/DIE GRÜNEN sowie der Abg.*“
- **Manual Annotation:**
  - `manual_label`: _[Sarcasm / Sincere_Positive / Ambiguous]_
  - `error_category`: _[Ironical_Agreement / Rhetorical_Politeness / Model_Misclassification]_
  - `notes`: _[]_

### 28. Interjection #119338: „Heidi Reichinnek [DIE LINKE]: Vielleicht mal zum Thema! Das wäre super!“
- **Sentiment Score:** `0.9772`
- **Interjection Faction (Opposition):** DIE LINKE
- **Main Speaker (Government):** Ruppert Stüwe (SPD)
- **Session:** Wahlperiode 20, Sitzung 126 (Date: 2023-09-29)
- **Context Before (5 sentences):** „*(Beifall bei der SPD sowie bei Abgeordneten des BÜNDNISSES 90/DIE GRÜNEN und der FDP) Vielen Dank, Herr Kollege Stüwe. – Das ist nun der Fluch der bösen Tat, wenn man Zwischenfragen nicht zulässt. Die AfD-Fraktion hat um eine Kurzintervention gebeten, die ich zulasse. – Bitte schön. (*“
- **Context After (3 sentences):** „*)*“
- **Manual Annotation:**
  - `manual_label`: _[Sarcasm / Sincere_Positive / Ambiguous]_
  - `error_category`: _[Ironical_Agreement / Rhetorical_Politeness / Model_Misclassification]_
  - `notes`: _[]_

### 29. Interjection #604174: „Robin Jünger [AfD]: Das ist gut! Dann hat’s gewirkt!“
- **Sentiment Score:** `0.9766`
- **Interjection Faction (Opposition):** AfD
- **Main Speaker (Government):** Carolin Wagner (SPD)
- **Session:** Wahlperiode 21, Sitzung 66 (Date: 2026-03-20)
- **Context Before (5 sentences):** „*– Frau Präsidentin! Werte Damen und Herren! Werte Zuhörerinnen und Zuhörer! Ich weiß nicht, wie es Ihnen geht, aber mir rast das Herz, wenn ich so eine Rede höre. (*“
- **Context After (3 sentences):** „*) Man sieht genau die Art und Weise der Politik der AfD: Ihnen soll Angst gemacht werden. (Alexander Arpaschi [AfD]: Nein, wir sagen die Wahrheit!) Sie sollen in Panik versetzt werden, damit Sie nicht mehr sehen und hören, was eigentlich im politischen Raum geordnet und reguliert wird. Sie sollen einfach nur aufgescheucht werden.*“
- **Manual Annotation:**
  - `manual_label`: _[Sarcasm / Sincere_Positive / Ambiguous]_
  - `error_category`: _[Ironical_Agreement / Rhetorical_Politeness / Model_Misclassification]_
  - `notes`: _[]_

### 30. Interjection #443205: „Gökay Akbulut [DIE LINKE]: Die brauchen Grundrenten! Respekt! Anerkennung!“
- **Sentiment Score:** `0.9764`
- **Interjection Faction (Opposition):** DIE LINKE
- **Main Speaker (Government):** Marian Wendt (CDU/CSU)
- **Session:** Wahlperiode 19, Sitzung 122 (Date: 2019-10-25)
- **Context Before (5 sentences):** „*Wir setzen im öffentlichen Dienst auf Eignung, Befähigung, fachliche Leistung. (Zuruf der Abg. Simone Barrientos [DIE LINKE]) Diese Bestenauslese können Sie übrigens auch im Grundgesetz – in Artikel 33 Absatz 2 – nachlesen. (Gökay Akbulut [DIE LINKE]: Sie haben Rentnerinnen und Rentner im Osten!) Mitleidsquoten brauchen die Menschen in Nord-, Süd-, Ost- oder Westdeutschland nicht. (*“
- **Context After (3 sentences):** „*) Meine Damen und Herren, wir wollen, dass die Mitarbeiter der Behörden des Bundes und deren nachgeordneter Verwaltungen so bunt sind wie unser Land. Deshalb haben wir in den letzten Jahren sehr erfolgreich neue Behörden oder Zweigstellen nicht nur in den Metropolen, sondern auch im ländlichen Raum Ostdeutschlands angesiedelt. Hier darf ich beispielsweise das BSI in Freital, das THW-Ausbildungszentrum in Brandenburg an der Havel, die Cyberagentur in meinem Wahlkreis in der Region Leipzig/Halle oder auch das Umweltbundesamt in Dessau-Roßlau erwähnen.*“
- **Manual Annotation:**
  - `manual_label`: _[Sarcasm / Sincere_Positive / Ambiguous]_
  - `error_category`: _[Ironical_Agreement / Rhetorical_Politeness / Model_Misclassification]_
  - `notes`: _[]_

### 31. Interjection #297130: „Friedrich Ostendorff [BÜNDNIS 90/DIE GRÜNEN]: Danke schön!“
- **Sentiment Score:** `0.9764`
- **Interjection Faction (Opposition):** BÜNDNIS 90/DIE GRÜNEN
- **Main Speaker (Government):** Christina Jantz (SPD)
- **Session:** Wahlperiode 18, Sitzung 161 (Date: 2016-03-17)
- **Context Before (5 sentences):** „*Warum macht ihr nichts? Ihr könnt doch was machen!) Dass jährlich rund 45 Millionen männliche Eintagsküken vergast werden, ist mit dem Staatsziel Tierschutz aus meiner Sicht nicht vereinbar. (Beifall bei Abgeordneten der SPD) Wenn männliche Küken für die Agrarindustrie unattraktiv werden, nur weil sie aus Legelinien stammen, dann stimmt hier etwas ganz grundsätzlich nicht. Ich möchte keinen Hehl daraus machen, dass die Anträge der Fraktion Bündnis 90/Die Grünen mit vielen unserer Positionen übereinstimmen. (*“
- **Context After (3 sentences):** „*) Beim Ziel, die Praxis der Tötung männlicher Eintagsküken zu beenden, weiß ich im Grunde genommen aber alle Bundestagsfraktionen vereint. (Beifall bei der SPD sowie bei Abgeordneten der CDU/CSU) Unterschiede tun sich entlang der Frage auf, wie dieses Ziel erreicht werden kann und mit welchem Nachdruck das Ziel verfolgt werden soll. (Britta Haßelmann [BÜNDNIS 90/DIE GRÜNEN]: Noch mal zwei Jahre warten!) Bundesminister Schmidt hat sich in dieser Frage deutlich positioniert.*“
- **Manual Annotation:**
  - `manual_label`: _[Sarcasm / Sincere_Positive / Ambiguous]_
  - `error_category`: _[Ironical_Agreement / Rhetorical_Politeness / Model_Misclassification]_
  - `notes`: _[]_

### 32. Interjection #207721: „Beatrix von Storch [AfD]: Ja, danke schön!“
- **Sentiment Score:** `0.9762`
- **Interjection Faction (Opposition):** AfD
- **Main Speaker (Government):** Heike Engelhardt (SPD)
- **Session:** Wahlperiode 20, Sitzung 211 (Date: 2025-01-31)
- **Context Before (5 sentences):** „*(Beifall bei der SPD sowie bei Abgeordneten des BÜNDNISSES 90/DIE GRÜNEN – Beatrix von Storch [AfD]: Was für eine Nazitruppe!) Sie haben hier jetzt mehrfach den Begriff „Nazis“ benutzt, (Beatrix von Storch [AfD]: Ja!) und Sie wissen, Frau Engelhardt, dass wir das hier rügen (Stephan Brandner [AfD]: „Rügen“? – Beatrix von Storch [AfD]: „Rügen“? Wir kriegen dafür Ordnungsrufe!) und zur Ordnung rufen. Insofern will ich ganz deutlich an dieser Stelle zur Ordnung rufen. (*“
- **Context After (3 sentences):** „*) Jetzt hat das Wort für die CDU/CSU-Fraktion Ralph Edelhäußer. (Beifall bei der CDU/CSU)*“
- **Manual Annotation:**
  - `manual_label`: _[Sarcasm / Sincere_Positive / Ambiguous]_
  - `error_category`: _[Ironical_Agreement / Rhetorical_Politeness / Model_Misclassification]_
  - `notes`: _[]_

### 33. Interjection #86601: „Jan Korte [DIE LINKE]: Gute Frage! Das ist eine gute Frage!“
- **Sentiment Score:** `0.9762`
- **Interjection Faction (Opposition):** DIE LINKE
- **Main Speaker (Government):** Katharina Beck (BÜNDNIS 90/DIE GRÜNEN)
- **Session:** Wahlperiode 20, Sitzung 97 (Date: 2023-04-20)
- **Context Before (5 sentences):** „*Olaf Scholzʼ Erinnerungslücken sind unglaubhaft, wir fordern endlich volle Transparenz! Dann gibt es hier eine schöne Social-Media-Kachel mit dem Gesicht Ihres Kollegen Herrn Schick drauf. Darauf steht – Zitat –: Scholz darf mit seiner Verschleierungstaktik nicht durchkommen. (Beifall bei der CDU/CSU sowie bei Abgeordneten der AfD und der LINKEN) Vor dem Hintergrund Ihrer Aussagen zu Herrn Schick würde mich Ihre Position dazu interessieren, ob Sie Herrn Schick zustimmen oder ob Sie das für falsch halten. (*“
- **Context After (3 sentences):** „*) Herr Schick spricht für sich selbst. (Kay Gottschalk [AfD]: Als Sachverständiger!) Er hat Finanzwende sogar mit gegründet. Und natürlich bin ich – das führe ich hier ja auch aus – extrem für Aufklärung.*“
- **Manual Annotation:**
  - `manual_label`: _[Sarcasm / Sincere_Positive / Ambiguous]_
  - `error_category`: _[Ironical_Agreement / Rhetorical_Politeness / Model_Misclassification]_
  - `notes`: _[]_

### 34. Interjection #563361: „Dr. Christina Baum [AfD]: Das ist ja gut! Ein guter Schritt!“
- **Sentiment Score:** `0.9758`
- **Interjection Faction (Opposition):** AfD
- **Main Speaker (Government):** Christos Pantazis (SPD)
- **Session:** Wahlperiode 21, Sitzung 24 (Date: 2025-09-17)
- **Context Before (5 sentences):** „*Frau Präsidentin! Verehrte Kolleginnen und Kollegen! Frau Baum, vorneweg, weil Sie eben die Thematik Hausärzte angesprochen haben: Ich darf Sie daran erinnern, dass wir noch in der Ampelkoalition die Entbudgetierung der Hausärzte miteinander besprochen und beschlossen haben. (*“
- **Context After (3 sentences):** „*) Deswegen ist der Sachverhalt so, wie Sie ihn dargestellt haben, rundweg falsch. (Beifall bei der SPD und der CDU/CSU – Dr. Paula Piechotta [BÜNDNIS 90/DIE GRÜNEN]: Aber nicht nur der!*“
- **Manual Annotation:**
  - `manual_label`: _[Sarcasm / Sincere_Positive / Ambiguous]_
  - `error_category`: _[Ironical_Agreement / Rhetorical_Politeness / Model_Misclassification]_
  - `notes`: _[]_

### 35. Interjection #280030: „Halina Wawzyniak [DIE LINKE]: Das ist schön!“
- **Sentiment Score:** `0.9756`
- **Interjection Faction (Opposition):** DIE LINKE
- **Main Speaker (Government):** Elisabeth Winkelmeier-Becker (CDU/CSU)
- **Session:** Wahlperiode 18, Sitzung 131 (Date: 2015-10-16)
- **Context Before (5 sentences):** „*(Zuruf von der LINKEN) Gerade nach der Anhörung der Sachverständigen am 21. September ist klar geworden, dass es sinnvoll gewesen wäre, in einigen Punkten auch über die jetzt vorgesehenen Regelungen hinauszugehen. Aber das, was wir jetzt als Kompromiss erarbeitet haben, stellt auf jeden Fall einen wichtigen Fortschritt dar. Wir haben uns bei der Sachverständigenanhörung genau erklären lassen, worum es geht. (*“
- **Context After (3 sentences):** „*) Es geht um schwere Kriminalität, bei der der Zugriff auf Verbindungsdaten erforderlich ist, um diese aufzuklären und um zu Verurteilungen kommen zu können. Aus der einjährigen Praxis eines Strafsenats beim BGH wurden uns viele Fälle geschildert, in denen Verkehrsdaten genau der entscheidende Ansatz waren, um Ermittlungen aufzunehmen und zu Beweisen zu kommen. Es ging in 20 Beispielsfällen aus einem Jahr um sieben Tötungsdelikte, (Halina Wawzyniak [DIE LINKE]: Alle ohne Vorratsdatenspeicherung angeklagt!) vier Raubdelikte, vier Bandendiebstähle, zwei Erpressungen, dazu um Betrug in Form des Enkeltricks, Brandstiftung und Betäubungsmittelkriminalität.*“
- **Manual Annotation:**
  - `manual_label`: _[Sarcasm / Sincere_Positive / Ambiguous]_
  - `error_category`: _[Ironical_Agreement / Rhetorical_Politeness / Model_Misclassification]_
  - `notes`: _[]_

### 36. Interjection #72722: „Hermann Färber [CDU/CSU]: Ja, danke schön!“
- **Sentiment Score:** `0.9755`
- **Interjection Faction (Opposition):** CDU/CSU
- **Main Speaker (Government):** Karlheinz Busen (FDP)
- **Session:** Wahlperiode 20, Sitzung 82 (Date: 2023-01-26)
- **Context Before (5 sentences):** „*Das schreibt mir Professor Bitter, der Präsident der Arbeitsgemeinschaft Deutscher Waldbesitzer. (Beifall bei der FDP, der SPD und dem BÜNDNIS 90/DIE GRÜNEN) Liebe Kolleginnen und Kollegen der CDU/CSU, so schlecht kann unser Programm gar nicht sein, wenn die Waldbesitzer dieses gut finden. Es ist vorhin schon gesagt worden: Wenn 7 000 Anträge eingegangen sind, dann muss das Programm doch irgendetwas Gutes haben. Lieber Kollege Färber, du hast viel Richtiges gesagt. (*“
- **Context After (3 sentences):** „*) Wir haben auch sehr lange gerungen, dass wir uns einig geworden sind, und natürlich gab es hier und da auch Unterschiede. Aber wir haben das nicht nach dem Motto gemacht: „Du bekommst hier etwas, dafür bekomme ich da etwas.“ Nein, im Gegenteil. Der Einzige, der etwas bekommt, das ist der Wald, und darum geht es.*“
- **Manual Annotation:**
  - `manual_label`: _[Sarcasm / Sincere_Positive / Ambiguous]_
  - `error_category`: _[Ironical_Agreement / Rhetorical_Politeness / Model_Misclassification]_
  - `notes`: _[]_

### 37. Interjection #325505: „Matthias W. Birkwald [DIE LINKE]: Das verstehen wir gut! Viel Erfolg!“
- **Sentiment Score:** `0.9753`
- **Interjection Faction (Opposition):** DIE LINKE
- **Main Speaker (Government):** Ansgar Heveling (CDU/CSU)
- **Session:** Wahlperiode 18, Sitzung 212 (Date: 2017-01-19)
- **Context Before (5 sentences):** „*Ich möchte zunächst Frau Staatsministerin Grütters entschuldigen. Aus Anlass des 75. Jahrestages der Wannsee-Konferenz findet jetzt eine Gedenkveranstaltung im Haus der Wannsee-Konferenz statt. Dort hält Frau Staatsministerin Grütters eine Rede in Vertretung der Bundeskanzlerin und bittet um Entschuldigung, dass sie heute nicht da sein kann. (*“
- **Context After (3 sentences):** „*) Frau Präsidentin! Meine liebe Kolleginnen und Kollegen! Meine Augen glänzten, als ich als generell historisch interessierter Schüler mit einem regen Interesse an Genealogie im Archiv des Erzbistums Paderborn auf ein Konvolut von Papieren stieß, das offensichtlich sämtliche Unterlagen, alle Rechnungen, Rechnungsbücher, Liegenschaftsaufzeichnungen, Korrespondenzen, interne Unterlagen zu Nutznießern, Gerichtsakten usw., angefangen von der Zeit des endenden 30-jährigen Krieges bis hinein ins 20.*“
- **Manual Annotation:**
  - `manual_label`: _[Sarcasm / Sincere_Positive / Ambiguous]_
  - `error_category`: _[Ironical_Agreement / Rhetorical_Politeness / Model_Misclassification]_
  - `notes`: _[]_

### 38. Interjection #298958: „Dr. André Hahn [DIE LINKE]: Gute! Gute Vorschläge!“
- **Sentiment Score:** `0.9753`
- **Interjection Faction (Opposition):** DIE LINKE
- **Main Speaker (Government):** Stephan Stracke (CDU/CSU)
- **Session:** Wahlperiode 18, Sitzung 165 (Date: 2016-04-15)
- **Context Before (5 sentences):** „*Mich erstaunt, wie viele Vorschläge hier neuerdings vonseiten der Bundesländer auf den Tisch gekommen sind. Ich glaube, im Rahmen der anstehenden Debatte müssen wir uns auf das Wesentliche konzentrieren und dürfen uns nicht mit der Beratung kleinteiliger Vorschläge verzetteln. Ich bin auf die entsprechende Sachverständigenanhörung gespannt, die wir in diesem Bereich vornehmen wollen. Interessant ist im Rahmen dieser Debatte, welche Vorschläge vonseiten der Opposition unterbreitet werden. (*“
- **Context After (3 sentences):** „*) Oftmals zielen diese Vorschläge darauf ab, die Zielrichtung des Gesetzes zu sprengen. Sie stellen häufig ein ganzes Sammelsurium dar. Entscheidend ist aber wohl die Frage der Sanktionen.*“
- **Manual Annotation:**
  - `manual_label`: _[Sarcasm / Sincere_Positive / Ambiguous]_
  - `error_category`: _[Ironical_Agreement / Rhetorical_Politeness / Model_Misclassification]_
  - `notes`: _[]_

### 39. Interjection #415412: „Peter Boehringer [AfD]: Das ist schön!“
- **Sentiment Score:** `0.9752`
- **Interjection Faction (Opposition):** AfD
- **Main Speaker (Government):** Andreas Nick (CDU/CSU)
- **Session:** Wahlperiode 19, Sitzung 89 (Date: 2019-03-21)
- **Context Before (5 sentences):** „*Nicht nur die politische und verfassungsrechtliche Krise dauert an, sondern auch die humanitäre Krise spitzt sich immer weiter zu. (Peter Boehringer [AfD]: Zur Sache!) Es ist daher mehr als zynisch, dass das Maduro-Regime die angebotene humanitäre Hilfe mit absurdesten Begründungen zurückgewiesen hat. Vor einigen Wochen konnte ich den Bundespräsidenten auf seiner Reise in die Nachbarländer Ecuador und Kolumbien begleiten. In Aufnahmeeinrichtungen, sowohl in Bogotá als auch in Quito, konnten wir uns im persönlichen Gespräch mit Flüchtlingen aus Venezuela über das Ausmaß der Krise im Land informieren. (*“
- **Context After (3 sentences):** „*) Inzwischen haben nach Angaben des UNHCR circa 3,4 Millionen Menschen Venezuela verlassen. Und ich sage es hier noch einmal ganz deutlich: Diese Krise hat deshalb auch das Potenzial, über Venezuela hinaus die gesamte Region zu destabilisieren. Dies alles bestätigt nochmals die Richtigkeit der Entscheidung der Bundesregierung und eines Großteils der EU-Staaten, Juan Guaidó als legitimen Übergangspräsidenten im Sinne der venezolanischen Verfassung anzuerkennen.*“
- **Manual Annotation:**
  - `manual_label`: _[Sarcasm / Sincere_Positive / Ambiguous]_
  - `error_category`: _[Ironical_Agreement / Rhetorical_Politeness / Model_Misclassification]_
  - `notes`: _[]_

### 40. Interjection #133302: „Peter Boehringer [AfD]: Das ist schön!“
- **Sentiment Score:** `0.9752`
- **Interjection Faction (Opposition):** AfD
- **Main Speaker (Government):** Rolf Mützenich (SPD)
- **Session:** Wahlperiode 20, Sitzung 139 (Date: 2023-11-28)
- **Context Before (5 sentences):** „*Wahnsinn!) Deswegen sage ich klar: Ja, wir müssen über einen verfassungsgemäßen Haushalt reden, auf der anderen Seite aber auch über unverdienten Reichtum in diesem Land. – Wir Sozialdemokratinnen und Sozialdemokraten werden in einer Woche auf dem Bundesparteitag (Dorothee Bär [CDU/CSU]: Ach du liebe Güte! Jetzt droht er mit seinem Parteitag!) die richtigen Antworten auf genau diese Herausforderungen geben, meine Damen und Herren. (Beifall bei der SPD – Peter Boehringer [AfD]: Das ist ein Fall für den Verfassungsschutz hier!) Es besteht kein Zweifel: Wir werden dieses Urteil umsetzen. (*“
- **Context After (3 sentences):** „*) Ich habe Respekt vor dem Bundesverfassungsgericht. (Peter Boehringer [AfD]: Nicht aus Respekt! Sie beugen sich!) Aber ich will gleichzeitig auch sagen: Ich hoffe, dass das Bundesverfassungsgericht auch vor anderen Verfassungsorganen wie dem Deutschen Bundestag Respekt hat.*“
- **Manual Annotation:**
  - `manual_label`: _[Sarcasm / Sincere_Positive / Ambiguous]_
  - `error_category`: _[Ironical_Agreement / Rhetorical_Politeness / Model_Misclassification]_
  - `notes`: _[]_

### 41. Interjection #322208: „Harald Petzold [Havelland] [DIE LINKE]: Sehr schön! – Renate Künast [BÜNDNIS 90/DIE GRÜNEN]: Das ist Ihren Ressorts ja wesensfremd!“
- **Sentiment Score:** `0.9752`
- **Interjection Faction (Opposition):** BÜNDNIS 90/DIE GRÜNEN
- **Main Speaker (Government):** Hendrik Hoppenstedt (CDU/CSU)
- **Session:** Wahlperiode 18, Sitzung 206 (Date: 2016-12-01)
- **Context Before (5 sentences):** „*Wir hatten zwei Themen, die uns beschäftigt haben, nämlich der Kreis der Anspruchsberechtigten zum einen und zum anderen die Anspruchshöhe. Beides haben wir in einem 30-minütigen Gespräch in Form eines Kompromisses beigelegt. Deswegen noch einmal: Für die Verzögerung ursächlich ist die Priorisierung im Hause Maas. Dort wird als Allererstes das aus dem Koalitionsvertrag abgearbeitet, was die SPD hineinverhandelt hat. (*“
- **Context After (3 sentences):** „*) Dann wird sich um die Dinge gekümmert, die im Zweifelsfall gar nicht – Frau Künast, hören Sie doch kurz zu! – im Koalitionsvertrag stehen, wie zum Beispiel die Reform des Mordparagrafen. Dann erst kommen unsere Belange.*“
- **Manual Annotation:**
  - `manual_label`: _[Sarcasm / Sincere_Positive / Ambiguous]_
  - `error_category`: _[Ironical_Agreement / Rhetorical_Politeness / Model_Misclassification]_
  - `notes`: _[]_

### 42. Interjection #547548: „Markus Frohnmaier [AfD]: Danke schön!“
- **Sentiment Score:** `0.9751`
- **Interjection Faction (Opposition):** AfD
- **Main Speaker (Government):** Johann David Wadephul (CDU/CSU)
- **Session:** Wahlperiode 21, Sitzung 9 (Date: 2025-06-04)
- **Context Before (5 sentences):** „*(Beifall bei der CDU/CSU, der SPD und dem BÜNDNIS 90/DIE GRÜNEN sowie bei Abgeordneten der Linken) Danke schön. 30 Sekunden. – Sie haben sich jetzt noch mal gemeldet. Das war gerade Ihre Nachfrage; insofern dürfen Sie jetzt nicht noch eine stellen. (*“
- **Context After (3 sentences):** „*) Danke. – Dann darf ich jetzt das Wort der CDU/CSU-Fraktion geben, an Herrn Abgeordneten Jürgen Hardt. (Stephan Brandner [AfD]: Völlig blanker Außenminister!)*“
- **Manual Annotation:**
  - `manual_label`: _[Sarcasm / Sincere_Positive / Ambiguous]_
  - `error_category`: _[Ironical_Agreement / Rhetorical_Politeness / Model_Misclassification]_
  - `notes`: _[]_

### 43. Interjection #623368: „Markus Frohnmaier [AfD]: Danke schön!“
- **Sentiment Score:** `0.9751`
- **Interjection Faction (Opposition):** AfD
- **Main Speaker (Government):** Derya Türk-Nachbaur (SPD)
- **Session:** Wahlperiode 21, Sitzung 86 (Date: 2026-06-25)
- **Context Before (5 sentences):** „*Sehr geehrte Frau Präsidentin! Die Rede eben hat wohl nach einer sehr patriotischen, sehr staatstragenden Rede klingen müssen. (*“
- **Context After (3 sentences):** „*) Sie war aber ein Geschenk an Moskau mit blau-brauner Schleife. (Beifall bei der SPD sowie bei Abgeordneten des BÜNDNISSES 90/DIE GRÜNEN und des Abg. Ulrich Thoden [Die Linke]) Ich glaube, Herr Putin wird sich erkenntlich zeigen.*“
- **Manual Annotation:**
  - `manual_label`: _[Sarcasm / Sincere_Positive / Ambiguous]_
  - `error_category`: _[Ironical_Agreement / Rhetorical_Politeness / Model_Misclassification]_
  - `notes`: _[]_

### 44. Interjection #517609: „Stefan Keuter [AfD]: Kommt! Danke schön!“
- **Sentiment Score:** `0.9747`
- **Interjection Faction (Opposition):** AfD
- **Main Speaker (Government):** Jens Spahn (CDU/CSU)
- **Session:** Wahlperiode 19, Sitzung 211 (Date: 2021-02-24)
- **Context Before (5 sentences):** „*Weil bei der Frage, welches Land man als Unternehmen in der Planung mit welcher Menge berücksichtigt, ein Vertrag mit der Bundesrepublik Deutschland als Absicherung sehr hilft. Die letzten drei Monate haben gezeigt, dass eine ausreichende Menge zur Verfügung steht. Bisher ist das noch nicht schlagend geworden, wie man sagen würde; bis jetzt sind alle Tests immer vollumfänglich im Markt gekauft und keine Tests derart beschafft worden. Deswegen müssen Sie mir im Zweifel die Information noch nachliefern, auf die Sie anspielen. (*“
- **Context After (3 sentences):** „*) – Daran habe ich keinen Zweifel. Vielen Dank, Herr Minister. – Die letzte Nachfrage in der Regierungsbefragung hat der Kollege Grigorios Aggelidis, FDP-Fraktion.*“
- **Manual Annotation:**
  - `manual_label`: _[Sarcasm / Sincere_Positive / Ambiguous]_
  - `error_category`: _[Ironical_Agreement / Rhetorical_Politeness / Model_Misclassification]_
  - `notes`: _[]_

### 45. Interjection #511136: „Zuruf von der AfD: Ein gutes!“
- **Sentiment Score:** `0.9746`
- **Interjection Faction (Opposition):** AfD
- **Main Speaker (Government):** Jens Spahn (CDU/CSU)
- **Session:** Wahlperiode 19, Sitzung 203 (Date: 2021-01-13)
- **Context Before (5 sentences):** „*Die deutsche Wirtschaft braucht offene Binnengrenzen und den freien Güterverkehr. Es ist eine Frage der ökonomischen Vernunft, dass wir nicht einzelne Nationen, sondern ganz Europa impfen. Erst so kommen wir wirtschaftlich wieder auf die Beine. (Beifall bei der CDU/CSU und der SPD sowie bei Abgeordneten des BÜNDNISSES 90/DIE GRÜNEN) Und ich frage auch all diejenigen, die den deutschen Alleingang bevorzugt hätten: Welches Signal hätten wir damit an unsere europäischen Partner gesandt? (*“
- **Context After (3 sentences):** „*) Hätten wir so das Vertrauen in Europa und in Deutschland gestärkt? Welche Konsequenzen hätte das für unsere Zusammenarbeit in der Zukunft, erst recht für Situationen, in denen wir mal auf Unterstützung und Zusammenarbeit angewiesen sind? In dieser Jahrhundertpandemie den europäischen Weg zu gehen, wird Europa stärken.*“
- **Manual Annotation:**
  - `manual_label`: _[Sarcasm / Sincere_Positive / Ambiguous]_
  - `error_category`: _[Ironical_Agreement / Rhetorical_Politeness / Model_Misclassification]_
  - `notes`: _[]_

### 46. Interjection #167663: „Sebastian Münzenmaier [AfD]: Aha! Das hat ja super funktioniert! Ganz großes Kino!“
- **Sentiment Score:** `0.9746`
- **Interjection Faction (Opposition):** AfD
- **Main Speaker (Government):** Isabel Cademartori Dujisin (SPD)
- **Session:** Wahlperiode 20, Sitzung 172 (Date: 2024-06-06)
- **Context Before (5 sentences):** „*Niemals sollte man sich dafür rechtfertigen müssen, dass man sich für die Gleichwertigkeit aller Menschen, für die universellen Menschenrechte, für das Völkerrecht und das friedliche Zusammenleben der Völker einsetzt. (Beifall bei der SPD sowie bei Abgeordneten des BÜNDNISSES 90/DIE GRÜNEN – Sebastian Münzenmaier [AfD]: Es geht nicht um Rhetorik! Es geht um Messerstecher! Verstehen Sie das nicht?) Auch in Mannheim gab es und gibt es keine falsche Toleranz gegenüber Islamisten. (*“
- **Context After (3 sentences):** „*) Das haben wir so auch in der „Mannheimer Erklärung“ für ein Zusammenleben in Vielfalt aufgeschrieben, die auch die muslimischen Gemeinschaften in unserer Stadt unterzeichnet haben. Darin steht, dass es keine Toleranz für diejenigen geben dürfe, deren Handeln sich gegen die Werte unseres Grundgesetzes richtet. (Sebastian Münzenmaier [AfD]: Na also!*“
- **Manual Annotation:**
  - `manual_label`: _[Sarcasm / Sincere_Positive / Ambiguous]_
  - `error_category`: _[Ironical_Agreement / Rhetorical_Politeness / Model_Misclassification]_
  - `notes`: _[]_

### 47. Interjection #524693: „Zuruf von der LINKEN: Einige ist gut! – Martin Reichardt [AfD]: Ganz schön viele!“
- **Sentiment Score:** `0.9744`
- **Interjection Faction (Opposition):** DIE LINKE
- **Main Speaker (Government):** Patrick Schnieder (CDU/CSU)
- **Session:** Wahlperiode 19, Sitzung 219 (Date: 2021-03-26)
- **Context Before (5 sentences):** „*Mitglied dieses Hohen Hauses zu sein, ist Ehre und Verpflichtung zugleich. Wer dieses Mandat ausüben darf, muss sich dabei allein am Nutzen für das Gemeinwohl orientieren. Haltung, Anstand und Integrität sind unverzichtbare Voraussetzungen für die Tätigkeit als Bundestagsabgeordneter, eine Tätigkeit im Übrigen – auch das muss man an dieser Stelle sagen –, die angemessen und vor allem ausreichend bezahlt ist. Für die ganz überwältigende Mehrheit der Kolleginnen und Kollegen der Unionsfraktion kann ich sagen, dass sie diesen Anspruch lebt und sich tagtäglich mit viel Einsatz im besten Sinne für das Wohl unseres Landes engagiert. Zur Wahrheit gehört aber auch, dass einige frühere Mitglieder unserer Fraktion (*“
- **Context After (3 sentences):** „*) diesem Anspruch nicht gerecht geworden sind und das Vertrauen, das Wählerinnen und Wähler in sie gesetzt haben, schwer enttäuscht haben. Dass die Menschen in unserem Land darüber empört sind, kann ich gut nachvollziehen. Ich kann Ihnen versichern: Mir und den anderen Mitgliedern meiner Fraktion geht es genauso.*“
- **Manual Annotation:**
  - `manual_label`: _[Sarcasm / Sincere_Positive / Ambiguous]_
  - `error_category`: _[Ironical_Agreement / Rhetorical_Politeness / Model_Misclassification]_
  - `notes`: _[]_

### 48. Interjection #439337: „Matthias W. Birkwald [DIE LINKE]: Ja, danke schön, damit es alle hören!“
- **Sentiment Score:** `0.9744`
- **Interjection Faction (Opposition):** DIE LINKE
- **Main Speaker (Government):** Frank Heinrich (CDU/CSU)
- **Session:** Wahlperiode 19, Sitzung 118 (Date: 2019-10-17)
- **Context Before (5 sentences):** „*Sehr geehrte Frau Präsidentin! Liebe Kolleginnen und Kollegen! Sie von den Linken bemängeln regelmäßig, wie es auch Kollegin Kolbe gesagt hat, (*“
- **Context After (3 sentences):** „*) dass die Rentenüberleitung nicht ausreichend sei und nicht sämtliche Besonderheiten des DDR-Rechts abdecke. Ich habe laut genickt, samt meinen Händen, als Herr Vogel in seiner Rede sagte, dass es tatsächlich verdächtig ist, dass diese Anträge so kurz vor der Wahl in Thüringen kommen. Das sollten die Wählerinnen und Wähler in Thüringen auch wissen, zumal Thüringen in der Statistik zur Rente mit am besten abschneidet.*“
- **Manual Annotation:**
  - `manual_label`: _[Sarcasm / Sincere_Positive / Ambiguous]_
  - `error_category`: _[Ironical_Agreement / Rhetorical_Politeness / Model_Misclassification]_
  - `notes`: _[]_

### 49. Interjection #351504: „Matthias W. Birkwald [DIE LINKE]: Ganz genau aufpassen, Herr Präsident! Danke schön!“
- **Sentiment Score:** `0.9742`
- **Interjection Faction (Opposition):** DIE LINKE
- **Main Speaker (Government):** Roy Kühne (CDU/CSU)
- **Session:** Wahlperiode 19, Sitzung 11 (Date: 2018-02-01)
- **Context Before (5 sentences):** „*– Wer ist dagegen? – Dann ist die Überweisung angenommen gegen die Stimmen der AfD-Fraktion. (Matthias W. Birkwald [DIE LINKE]: Nein, der Linken!) – Gegen die Stimmen der Fraktion Die Linke. (*“
- **Context After (3 sentences):** „*) – Entschuldigung. (Matthias W. Birkwald [DIE LINKE]: Und die Grünen!) – Und gegen die Stimmen der Grünen.*“
- **Manual Annotation:**
  - `manual_label`: _[Sarcasm / Sincere_Positive / Ambiguous]_
  - `error_category`: _[Ironical_Agreement / Rhetorical_Politeness / Model_Misclassification]_
  - `notes`: _[]_

### 50. Interjection #284203: „Renate Künast [BÜNDNIS 90/DIE GRÜNEN]: Oh ja, schön!“
- **Sentiment Score:** `0.9742`
- **Interjection Faction (Opposition):** BÜNDNIS 90/DIE GRÜNEN
- **Main Speaker (Government):** Metin Hakverdi (SPD)
- **Session:** Wahlperiode 18, Sitzung 138 (Date: 2015-11-24)
- **Context Before (5 sentences):** „*Der Ausbau des Verbraucherschutzes muss aber weitergehen. Die Ausweitung des Anwendungsbereichs des Unterlassungsklagegesetzes stellt aus meiner Sicht einen wichtigen Baustein dar. Deshalb will ich meinen Appell an die Kolleginnen und Kollegen von der Union hier erneuern: Bitte geben Sie sich einen Ruck, damit wir in diesem Punkt endlich zu einem Gesetz kommen. – Ein weiteres wichtiges Gesetzesvorhaben stellt aus meiner Sicht die Musterfeststellungsklage dar. (*“
- **Context After (3 sentences):** „*) Wir haben über diese bereits gesprochen. Wir freuen uns auf den Entwurf des Ministeriums. (Beifall bei der SPD) Der zweite Schwerpunkt, der in diesem Etat gesetzt wird, betrifft aus meiner Sicht die Stärkung des Rechtsstaates.*“
- **Manual Annotation:**
  - `manual_label`: _[Sarcasm / Sincere_Positive / Ambiguous]_
  - `error_category`: _[Ironical_Agreement / Rhetorical_Politeness / Model_Misclassification]_
  - `notes`: _[]_
