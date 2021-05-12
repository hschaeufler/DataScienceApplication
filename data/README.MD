Text übernommen von https://npgeo-corona-npgeo-de.hub.arcgis.com/datasets/dd4580c810204019a7b8eb3e0b329dd6_0

Quelle: Robert Koch Institut


    Dem Dashboard liegen aggregierte Daten der gemäß IfSG von den Gesundheitsämtern an das RKI übermittelten Covid-19-Fälle zu Grunde
    Mit den Daten wird der tagesaktuelle Stand (00:00 Uhr) abgebildet und es werden die Veränderungen bei den Fällen und Todesfällen zum Vortag dargstellt
    In der Datenquelle sind folgende Parameter enthalten:
        IdBundesland: Id des Bundeslands des Falles mit 1=Schleswig-Holstein bis 16=Thüringen
        Bundesland: Name des Bundeslanes
        Landkreis ID: Id des Landkreises des Falles in der üblichen Kodierung 1001 bis 16077=LK Altenburger Land
        Landkreis: Name des Landkreises
        Altersgruppe: Altersgruppe des Falles aus den 6 Gruppe 0-4, 5-14, 15-34, 35-59, 60-79, 80+ sowie unbekannt
        Altersgruppe2: Altersgruppe des Falles aus 5-Jahresgruppen 0-4, 5-9, 10-14, ..., 75-79, 80+ sowie unbekannt
        Geschlecht: Geschlecht des Falles M0männlich, W=weiblich und unbekannt
        AnzahlFall: Anzahl der Fälle in der entsprechenden Gruppe
        AnzahlTodesfall: Anzahl der Todesfälle in der entsprechenden Gruppe
        Meldedatum: Datum, wann der Fall dem Gesundheitsamt bekannt geworden ist
        Datenstand: Datum, wann der Datensatz zuletzt aktualisiert worden ist
        NeuerFall: 
            0: Fall ist in der Publikation für den aktuellen Tag und in der für den Vortag enthalten
            1: Fall ist nur in der aktuellen Publikation enthalten
            -1: Fall ist nur in der Publikation des Vortags enthalten
            damit ergibt sich: Anzahl Fälle der aktuellen Publikation als Summe(AnzahlFall), wenn NeuerFall in (0,1); Delta zum Vortag als Summe(AnzahlFall) wenn NeuerFall in (-1,1)
        NeuerTodesfall:
            0: Fall ist in der Publikation für den aktuellen Tag und in der für den Vortag jeweils ein Todesfall
            1: Fall ist in der aktuellen Publikation ein Todesfall, nicht jedoch in der Publikation des Vortages
            -1: Fall ist in der aktuellen Publikation kein Todesfall, jedoch war er in der Publikation des Vortags ein Todesfall
            -9: Fall ist weder in der aktuellen Publikation noch in der des Vortages ein Todesfall
            damit ergibt sich: Anzahl Todesfälle der aktuellen Publikation als Summe(AnzahlTodesfall) wenn NeuerTodesfall in (0,1); Delta zum Vortag als Summe(AnzahlTodesfall) wenn NeuerTodesfall in (-1,1)
        Referenzdatum: Erkrankungsdatum bzw. wenn das nicht bekannt ist, das Meldedatum
        AnzahlGenesen: Anzahl der Genesenen in der entsprechenden Gruppe
        NeuGenesen:
            0: Fall ist in der Publikation für den aktuellen Tag und in der für den Vortag jeweils Genesen
            1: Fall ist in der aktuellen Publikation Genesen, nicht jedoch in der Publikation des Vortages
            -1: Fall ist in der aktuellen Publikation nicht Genesen, jedoch war er in der Publikation des Vortags Genesen
            -9: Fall ist weder in der aktuellen Publikation noch in der des Vortages Genesen 
            damit ergibt sich: Anzahl Genesen der aktuellen Publikation als Summe(AnzahlGenesen) wenn NeuGenesen in (0,1); Delta zum Vortag als Summe(AnzahlGenesen) wenn NeuGenesen in (-1,1)
        IstErkrankungsbeginn: 1, wenn das Refdatum der Erkrankungsbeginn ist, 0 sonst
