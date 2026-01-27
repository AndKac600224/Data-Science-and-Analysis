
## Projekt dotyczący metod klasyfikacji zbioru danych Student Performance

Kacper Andrzejewski 419925, gr. 1, IAD rok 3 AGH

#### Projekt polega na zastosowaniu metod uczenia maszynowego (uczenie nienadzorowane) w celu klasyfikacji danych dotyczących zbioru Student Performance zawierającego 10000 rekordów (danych na temat różnych studentów) oraz kolumny - 5 zmiennych niezależnych:

- **Hours Studied** - ilość godzin poświęcona na naukę przez studenta,
- **Previous Scores** - wyniki uzyskane przez studenta w poprzednich
  testach
- **Extracurricular Activities** - zmienna binarna określająca, czy
  student bierze udział w dodatkowych aktywnościach (teoretycznie
  podnoszących jego poziom wiedzy),
- **Sleep Hours** - średnia liczba godzin snu studenta,
- **Sample Question Papers Practiced** - ilość wykonanych przykładowych
  testów przez studenta

#### Oraz zmienną zależną, determinowaną przez powyższe:

- **Performance Index** - indeks określający miarę wyników studenta w
  przedziale od 10 do 100, gdzie im wyższa wartość, tym lepsze jego
  rezultaty.

Wszystkie zmienne (oprócz *Extracurricular Activities*) są zmiennymi
numerycznymi, więc, poza wspomnianą zmienną binarną, nie ma potrzeby
konwersji w celu przygotowania ich do użycia w przyszłych modelach.

### Etap 1 - wczytanie, preprocessing i wgląd do danych

``` r
data_in <- read.csv("Student_Performance.csv", header=TRUE, sep=',')
head(data_in, 10)
```

    ##    Hours.Studied Previous.Scores Extracurricular.Activities Sleep.Hours
    ## 1              7              99                        Yes           9
    ## 2              4              82                         No           4
    ## 3              8              51                        Yes           7
    ## 4              5              52                        Yes           5
    ## 5              7              75                         No           8
    ## 6              3              78                         No           9
    ## 7              7              73                        Yes           5
    ## 8              8              45                        Yes           4
    ## 9              5              77                         No           8
    ## 10             4              89                         No           4
    ##    Sample.Question.Papers.Practiced Performance.Index
    ## 1                                 1                91
    ## 2                                 2                65
    ## 3                                 2                45
    ## 4                                 2                36
    ## 5                                 5                66
    ## 6                                 6                61
    ## 7                                 6                63
    ## 8                                 6                42
    ## 9                                 2                61
    ## 10                                0                69

Powyżej wyświetlone zostało 10 pierwszych wierszy w zbiorze danych.

``` r
sum(is.na(data_in))
```

    ## [1] 0

Zgodnie z wynikiem powyższego kodu, w zbiorze danych nie ma brakujących
wartości (*Missing Values* - *NA*), więc nie ma potrzeby usuwania
pojedynczych wierszy bądź dokonywania imputowania.

``` r
summary(data_in)
```

    ##  Hours.Studied   Previous.Scores Extracurricular.Activities  Sleep.Hours   
    ##  Min.   :1.000   Min.   :40.00   Length:10000               Min.   :4.000  
    ##  1st Qu.:3.000   1st Qu.:54.00   Class :character           1st Qu.:5.000  
    ##  Median :5.000   Median :69.00   Mode  :character           Median :7.000  
    ##  Mean   :4.993   Mean   :69.45                              Mean   :6.531  
    ##  3rd Qu.:7.000   3rd Qu.:85.00                              3rd Qu.:8.000  
    ##  Max.   :9.000   Max.   :99.00                              Max.   :9.000  
    ##  Sample.Question.Papers.Practiced Performance.Index
    ##  Min.   :0.000                    Min.   : 10.00   
    ##  1st Qu.:2.000                    1st Qu.: 40.00   
    ##  Median :5.000                    Median : 55.00   
    ##  Mean   :4.583                    Mean   : 55.22   
    ##  3rd Qu.:7.000                    3rd Qu.: 71.00   
    ##  Max.   :9.000                    Max.   :100.00

Powyżej wyświetlono podstawowe informacje statystyczne o każdej zmiennej
(minima, maxima, mediany, wartości oczekiwane, pierwsze oraz trzecie
kwartyle).

``` r
library(dplyr)
data<- data_in %>%
  mutate(Extracurricular.Activities = ifelse(Extracurricular.Activities == "Yes", 1, 0)
         )
head(data, 10)
```

    ##    Hours.Studied Previous.Scores Extracurricular.Activities Sleep.Hours
    ## 1              7              99                          1           9
    ## 2              4              82                          0           4
    ## 3              8              51                          1           7
    ## 4              5              52                          1           5
    ## 5              7              75                          0           8
    ## 6              3              78                          0           9
    ## 7              7              73                          1           5
    ## 8              8              45                          1           4
    ## 9              5              77                          0           8
    ## 10             4              89                          0           4
    ##    Sample.Question.Papers.Practiced Performance.Index
    ## 1                                 1                91
    ## 2                                 2                65
    ## 3                                 2                45
    ## 4                                 2                36
    ## 5                                 5                66
    ## 6                                 6                61
    ## 7                                 6                63
    ## 8                                 6                42
    ## 9                                 2                61
    ## 10                                0                69

Powyżej zmienna jakościowa (binarna) *Extracurricular Activities*
została przetransformowana na wartości typu float - “No” oznacza 0,
natomiast “Yes” odpowiada 1 (binarna).

``` r
num_data <- data[,c(1,2,4,5,6)]
par(mfrow=c(2,3))
for (i in 1:ncol(num_data)) {
  boxplot(num_data[, i],
          main = paste("Wykres boxplot:", names(num_data)[i]),
          cex.main=0.77,
          col='lightgreen',
          ylab='Wartość',
          outpch=19,
          outcol='red')
}
```

![](projekt_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

Powyżej zostały wyświetlone wykresy typu ramka-wąsy dla wszystkich
zmiennych numerycznych (niebinarnych), aby przedstawić graficznie
rozkład wartości w każdej kolumnie. Wartości odstające (tzw. *Outliers*)
powinny być oznaczone jako czerwone punkty, lecz brakuje ich na
wykresach, co oznacza, że w danych nie występują wartości tego typu. Dla
pewności oraz dokładniejszego zobrazowania rozkładu wartości zmiennych
poniżej wykonano histogramy tych samych zmiennych. Ilość przedziałow
histogramu została wyznaczona za pomocą *reguły Scotta*, ponieważ jest
ona odpowiednim wyborem do tego zbioru danych, gdyż nie posiada on
wartości odstających oraz ma dużą ilość rekordów (10000).

``` r
par(mfrow=c(2,3))
for (i in 1:ncol(num_data)) {
  hist(num_data[, i],
          main = paste("Histogram:", names(num_data)[i]),
          breaks="Scott",
          cex.main=0.77,
          col='lightblue',
          ylab='Wartość',
          xlab = names(num_data)[i])
}
```

![](projekt_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

Jak widać powyżej, niektóre zmienne ze względu na istotę swoich wartości
mają histogramy z oddzielonymi słupkami, lecz wynika to z natury danych
zmiennych oraz zastosowania odpowiedniej reguły podziału wykresu, aby
poprawnie przedstawić zmienne *Previous Scores* oraz *Performance
Index*, które mają ciągły charakter histogramu. Ponownie potwierdzono
tezę, iż w danych nie występują wartości odstające.

### Etap 2 - Przygotowanie danych do wykonania modeli

``` r
set.seed(419925)
sets <- sample(1:nrow(data), 0.7 * nrow(data))
train_raw <- data[sets, ]
temp_raw <- data[-sets, ]

random4 <- sample(1:nrow(temp_raw), 4)
val_raw <- temp_raw[-random4, ]
test_raw <- temp_raw[random4, ]
```

Powyżej został dokonany podział zbioru danych na treningowy (70% zbioru
danych), testowy do predykcji metodą k-najbliższych sąsiadów (4 losowe
obserwacje) oraz walidacyjny do oceny skuteczności na zbiorze (pozostałe
30% bez 4 testowych obserwacji).

``` r
stand <- function(x) { (x - mean(x)) / sd(x) }
train_std <- train_raw
train_std[, 1:5] <- as.data.frame(lapply(train_raw[, 1:5], stand))

train_means <- sapply(train_raw[, 1:5], mean)
train_sds <- sapply(train_raw[, 1:5], sd)
val_std <- val_raw
test_std <- test_raw

for(i in 1:5) {
  val_std[, i] <- (val_raw[, i] - train_means[i]) / train_sds[i]
  test_std[, i] <- (test_raw[, i] - train_means[i]) / train_sds[i]
}
```

``` r
num_train_std <- train_std[,c(1,2,4,5,6)]
par(mfrow=c(2,3))
for (i in 1:ncol(num_train_std)) {
  boxplot(num_train_std[, i],
          main = paste("Wykres boxplot:", names(num_train_std)[i]),
          cex.main=0.77,
          col='lightgreen',
          ylab='Wartość',
          outpch=19,
          outcol='red')
}
```

![](projekt_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

``` r
summary(train_std)
```

    ##  Hours.Studied      Previous.Scores    Extracurricular.Activities
    ##  Min.   :-1.53798   Min.   :-1.69702   Min.   :-0.9942           
    ##  1st Qu.:-0.76301   1st Qu.:-0.88775   1st Qu.:-0.9942           
    ##  Median : 0.01196   Median :-0.02068   Median :-0.9942           
    ##  Mean   : 0.00000   Mean   : 0.00000   Mean   : 0.0000           
    ##  3rd Qu.: 0.78692   3rd Qu.: 0.90420   3rd Qu.: 1.0057           
    ##  Max.   : 1.56189   Max.   : 1.71347   Max.   : 1.0057           
    ##   Sleep.Hours      Sample.Question.Papers.Practiced Performance.Index
    ##  Min.   :-1.4856   Min.   :-1.5891                  Min.   : 10.00   
    ##  1st Qu.:-0.8974   1st Qu.:-0.8891                  1st Qu.: 40.00   
    ##  Median : 0.2790   Median : 0.1609                  Median : 55.00   
    ##  Mean   : 0.0000   Mean   : 0.0000                  Mean   : 55.06   
    ##  3rd Qu.: 0.8672   3rd Qu.: 0.8609                  3rd Qu.: 70.00   
    ##  Max.   : 1.4554   Max.   : 1.5609                  Max.   :100.00

Powyżej dokonano standaryzacji zmiennych niezależnych zbioru
treningowego oraz wyliczono metryki danych w tym zbiorze (średnia i
odchylenie standardowe), aby użyć ich do standaryzacji zmiennych
niezależnych zbioru walidacyjnego oraz testowego. Dzięki temu wartości
są zeskalowane tak, aby różnice między nimi nie były znaczące (w sensie
bezwzględnym). Pominięcie tej czynności spowodowałoby faworyzowanie
przez przyszłe modele klasyfikacyjne zmiennych o wysokich wartościach.
Modele klasyfikacyjne bazują na odległościach między wartościami
zmiennych w przestrzeni, więc po przeskalowaniu tych wartości, wszystkie
zmienne będą traktowane przez model w równym stopniu, niezależnie od
swoich wielkości. Powyżej dla podglądu rezultatów wykonano ponownie
wykresy Boxplot dla zmiennych niezależnych niebinarnych zbioru
treningowego oraz wyświetlono ich podstawowe statystyki.

### Etap 3 - Hierarchiczna analiza skupień - metoda aglomeracyjna

#### Metryka euklidesowa

``` r
dist_e <- dist(train_std[, 1:5], method='euclidean')
m1_e <- hclust(dist_e, method='average')
m2_e <- hclust(dist_e, method='complete')
m3_e <- hclust(dist_e, method='ward.D')
m4_e <- hclust(dist_e, method='single')
```

``` r
par(mfrow=c(1,2))
plot(m1_e, main="Method: Average") 
plot(m2_e, main="Method: Complete")
```

![](projekt_files/figure-gfm/unnamed-chunk-12-1.png)<!-- -->

``` r
par(mfrow=c(1,2))
plot(m3_e, main="Method: Ward.D") 
plot(m4_e, main="Method: Single") 
```

![](projekt_files/figure-gfm/unnamed-chunk-13-1.png)<!-- -->

Po analizie jakościowej wykresów (dendrogramów) dla 4 różnych metod
liczenia dystansów (*single*, *average*, *complete*, *ward.D*) można
wysunąć następujące wnioski:  
\* **Metoda Average** - optymalną wartością dla k jest 2, ponieważ dla
dalszych podgrup klastry są coraz mniej zwarte i coraz bliżej siebie,  
\* **Metoda Complete** - optymalną wartością dla k jest 3, ponieważ
zauważalny jest ostatni wyraźny podział głównych gałęzi na corazto
mniejsze podgrupy,  
\* **Metoda Ward.D** - optymalną wartością dla k jest 2, ponieważ
większe wartości doprowadziłyby do coraz mniejszych różnicach między
klastrami (zatracenie informacji). Metoda ta wykazuje się najlepszą
spośród wszystkich, ponieważ ma dwie bardzo długie gałęzie na początku
co, w odróżnieniu do pozostałych metod, świadczy o najlepszym podziale
danych pod względem zbalansowania podgrup oraz wydzielenia różnic między
nimi,  
\* **Metoda Single** - brak optymalnej wartości k, ponieważ ta technika
liczenia dystansu prowadzi do tzw. *efektu łańcuchowania* tworząc
pojedyncze gałęzie i brak zbalansowanego wydzielenia klastrów.

Do dalszej analizy hierarchicznej analizy skupień za pomocą metryki
euklidesowej uwzględnione zostaną 3 pierwsze wymienione metody spośród
powyższych.

``` r
cut_m1_e <- cutree(m1_e, k=2)
cut_m2_e <- cutree(m2_e, k=3)
cut_m3_e <- cutree(m3_e, k=2)
```

Powyżej nastąpiło *obcięcie* dendrogramu na ustalonych wysokościach
(zgodnie z wysuniętymi poprzednio wnioskami).

``` r
library(cluster)
library(clValid)

dunn(dist_e, cut_m1_e)
```

    ## [1] 0.3169535

``` r
dunn(dist_e, cut_m2_e)
```

    ## [1] 0.02897083

``` r
dunn(dist_e, cut_m3_e)
```

    ## [1] 0.3169535

Dla wspomnianych metod liczenia dystansu obliczono *indeks Dunna*, który
jest miarą “jakości” podziału na grupy w zbiorze. Jest to stosunek
minimalnej odległości między klastrami do maksymalnej średnicy klastra.
Zatem im wyższa wartość, tym lepszy podział. Najwyższą wartością
wykazały się wyniki dla metod *Average* oraz *Ward.D* osiągając tą samą
wartość (0.317). Wielkość dla metody *Complete* jest o rząd niższa od
pozostałych, więc metoda ta nie jest odpowiednim wyborem.

``` r
print("---------------------Ward.D----------------------")
```

    ## [1] "---------------------Ward.D----------------------"

``` r
sil_ward_e <- silhouette(cut_m3_e, dist_e)
summary(sil_ward_e) 
```

    ## Silhouette of 7000 units in 2 clusters from silhouette.default(x = cut_m3_e, dist = dist_e) :
    ##  Cluster sizes and average silhouette widths:
    ##      3520      3480 
    ## 0.2074291 0.2113621 
    ## Individual silhouette widths:
    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##  0.1268  0.1855  0.2070  0.2094  0.2303  0.3071

``` r
cat("\n---------------------Complete--------------------\n")
```

    ## 
    ## ---------------------Complete--------------------

``` r
sil_complete_e <- silhouette(cut_m2_e, dist_e)
summary(sil_complete_e)
```

    ## Silhouette of 7000 units in 3 clusters from silhouette.default(x = cut_m2_e, dist = dist_e) :
    ##  Cluster sizes and average silhouette widths:
    ##       3026       1457       2517 
    ## 0.05951590 0.15060066 0.05519216 
    ## Individual silhouette widths:
    ##      Min.   1st Qu.    Median      Mean   3rd Qu.      Max. 
    ## -0.277357  0.002728  0.088369  0.076920  0.160264  0.350081

``` r
cat("\n---------------------Average---------------------")
```

    ## 
    ## ---------------------Average---------------------

``` r
sil_average_e <- silhouette(cut_m1_e, dist_e)
summary(sil_average_e)
```

    ## Silhouette of 7000 units in 2 clusters from silhouette.default(x = cut_m1_e, dist = dist_e) :
    ##  Cluster sizes and average silhouette widths:
    ##      3520      3480 
    ## 0.2074291 0.2113621 
    ## Individual silhouette widths:
    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##  0.1268  0.1855  0.2070  0.2094  0.2303  0.3071

Powyżej został wyliczony indeks Silhouette dla wspomnianych trzech
sposobów liczenia dystansów. Stanowi on miarę tego, jak dobrze
poszczególne obserwacje pasują do przypisanych im klastrów (im wyższa
wartość, tym lepsza separacja).  
Analizując uzyskane wyniki, można wysunąć następujące wnioski:  
\* **Metody Ward.D** oraz **Average** uzyskały identyczne wyniki,
osiągając najwyższą średnią wartość indeksu Silhouette na poziomie
0.2094. W obu przypadkach algorytmy wyodrębniły dwa bardzo zbliżone
liczebnie klastry (3520 oraz 3480 jednostek). Fakt, że tak różne
matematycznie podejścia do łączenia grup dały ten sam rezultat, świadczy
o niezwykle silnej i naturalnej strukturze danych przy podziale na k=2.
Średnie szerokości sylwetki w obu klastrach oscylują wokół wartości
0.21, co wskazuje na stabilne przyporządkowanie większości obserwacji,  
\* **Metoda Complete** wykazała się znacznie słabszymi wynikami. Średnia
wartość indeksu Silhouette wyniosła dla niej jedynie 0.076. Ponadto,
minimalne wartości szerokości sylwetki dążą do poziomów ujemnych
(-0.277), co sugeruje, że wiele punktów mogło zostać błędnie
zaklasyfikowanych lub znajduje się na bardzo niepewnych granicach między
klastrami.

Uzyskane parametry jednoznacznie potwierdzają, że optymalnym wyborem dla
hierarchicznej analizy skupień w tym zbiorze danych (metryka
euklidesowa) jest podział na k=2 klastry. Wynik 0.2094 jest niemal
trzykrotnie lepszy niż w przypadku metody **Complete**, a zbieżność
wyników metod **Ward.D** i **Average** stanowi silny dowód na
obiektywność i trwałość takiego podziału.

``` r
results_clus_e <- mutate(train_std[, 1:5], m1_e=cut_m3_e)
table(results_clus_e$m1_e)
```

    ## 
    ##    1    2 
    ## 3520 3480

Powyżej wyświetlono liczebność punktów we wszystkich stworzonych
klastrach. Podziały są zbalansowane i zrównoważone, co oznacza, że
wybrane k=2 klastrów stanowi silny i odpowiedni podział danych.

#### Metryka Manhattan

``` r
dist_m <- dist(train_std[, 1:5], method='manhattan')
m1_m <- hclust(dist_m, method='average')
m2_m <- hclust(dist_m, method='complete')
m3_m <- hclust(dist_m, method='ward.D')
m4_m <- hclust(dist_m, method='single')
```

Dla porównania wykonano hierarchiczną analizę skupień, lecz dla metryki
Manhattan. Wykorzystano wszystkie 4 użyte poprzednio metody wyliczania
dystansów.

``` r
par(mfrow=c(1,2))
plot(m1_m, main="Method: Average") 
plot(m2_m, main="Method: Complete")
```

![](projekt_files/figure-gfm/unnamed-chunk-19-1.png)<!-- -->

``` r
par(mfrow=c(1,2))
plot(m3_m, main="Method: Ward.D") 
plot(m4_m, main="Method: Single") 
```

![](projekt_files/figure-gfm/unnamed-chunk-20-1.png)<!-- -->

Na podstawie analizy jakościowej stworzony dendrogramów można wysunąć
bardzo podobne wnioski, jak w przypadku metryki euklidesowej.
Najbardziej zróżnicowanym i zbalansowanym podziałem charakteryzuje się
**metoda Ward.D** (optymalną ilością klastrów jest ponownie k=2).
**Metoda Average** i **metoda Complete** ponownie przedstawiają o wiele
mniejsze różnice pomiędzy wydzielonymi podgrupami, natomiast klastry są
umiarkowanie dobrze zbalansowane i dla odpowiednio k=2 i k=3 (dla
**metody Average** i **metody Complete**) można wydzielić odpowiednio
zwarte grupy (wybranie większej wartości k skutkowałoby zatraceniem
różnic pomiędzy podziałami i zbyt szczegółowy podział - *“na siłę”*).
Ostatnia metoda - **Single** ponownie jest nieodpowiednią techniką
wyliczania dystansów, ponieważ dendrogram charakteryzuje się
niepożądanym *efektem łańcuchowym*, co sprawia, iż podgrupy nie są
poprawnie stworzone.

``` r
cut_m1_m <- cutree(m1_m, k=2)
cut_m2_m <- cutree(m2_m, k=3)
cut_m3_m <- cutree(m3_m, k=2)
```

Powyżej dokonano *przycięcia* dendrogramów dla ustalonej dla każdej
metody wartości k.

``` r
dunn(dist_m, cut_m1_m)
```

    ## [1] 0.1587042

``` r
dunn(dist_m, cut_m2_m)
```

    ## [1] 0.0186917

``` r
dunn(dist_m, cut_m3_m)
```

    ## [1] 0.1587042

Dla metryki Manhattan wartości indeksu Dunna są zdecydowanie niższe niż
dla poprzedniej. Oznacza to, iż spośród tych 2 metod bardziej optymalną
(pod względem doboru na podstawie indeksu Dunna) jest metryka
euklidesowa. Mimo ponownego wykazania się największą najwyższą wartością
indeksu Dunna dla **metody Ward.D** oraz **Average** (metryka
**Manhattan** - 0.1587) spośród wyliczonych 3, jest ona około 2 razy
niższa niż ta sama statystyka dla metryki euklidesowej (ok. 0.31).

``` r
print("---------------------Ward.D----------------------")
```

    ## [1] "---------------------Ward.D----------------------"

``` r
sil_ward_m <- silhouette(cut_m3_m, dist_m)
summary(sil_ward_m) 
```

    ## Silhouette of 7000 units in 2 clusters from silhouette.default(x = cut_m3_m, dist = dist_m) :
    ##  Cluster sizes and average silhouette widths:
    ##      3520      3480 
    ## 0.3033970 0.3070876 
    ## Individual silhouette widths:
    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##  0.2368  0.2885  0.3048  0.3052  0.3212  0.3699

``` r
cat("\n---------------------Complete--------------------\n")
```

    ## 
    ## ---------------------Complete--------------------

``` r
sil_complete_m <- silhouette(cut_m2_m, dist_m)
summary(sil_complete_m)
```

    ## Silhouette of 7000 units in 3 clusters from silhouette.default(x = cut_m2_m, dist = dist_m) :
    ##  Cluster sizes and average silhouette widths:
    ##       2685        835       3480 
    ## 0.08164768 0.24139086 0.25617062 
    ## Individual silhouette widths:
    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ## -0.3409  0.1101  0.2431  0.1875  0.2896  0.4187

``` r
cat("\n---------------------Average---------------------")
```

    ## 
    ## ---------------------Average---------------------

``` r
sil_average_m <- silhouette(cut_m1_m, dist_m)
summary(sil_average_m)
```

    ## Silhouette of 7000 units in 2 clusters from silhouette.default(x = cut_m1_m, dist = dist_m) :
    ##  Cluster sizes and average silhouette widths:
    ##      3520      3480 
    ## 0.3033970 0.3070876 
    ## Individual silhouette widths:
    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##  0.2368  0.2885  0.3048  0.3052  0.3212  0.3699

W porównaniu do metryki euklidesowej, indeks Silhouette okazał się
znacznie wyższy w tym przypadku (0.3052 dla **metody Ward.D** oraz
**metody Average**). Mimo że metryka euklidesowa z **metodą Ward.D**
dała najwyższy indeks Dunna (0.31), co sugeruje dobrą separację
skrajnych grup, zdecydowano się na wybór metryki **Manhattan z metodą
Warda**. Przemawia za tym znacznie wyższa średnia szerokość sylwetki
(*Silhouette* = 0.31 vs 0.2094), co wskazuje na lepsze dopasowanie
większości obserwacji do przypisanych im klastrów i mniejsze ryzyko
błędnego zaklasyfikowania studentów znajdujących się na granicach grup.
Indeks *Silhouette* w problemie klasyfikacji hierarchiczną analizą
skupień dla treningowego zbioru danych wykazuje się w analizie wyższym
priorytetem niż indeks *Dunna*, który pozwala jedynie na wstępny pogląd
jakości podziałów.

``` r
results_clus_m <- mutate(train_std[, 1:5], m1_m=cut_m3_m)
table(results_clus_m$m1_m)
```

    ## 
    ##    1    2 
    ## 3520 3480

Dla wybranej metryki oraz metody liczenia dystansów przedstawiono
powyżej liczebność danych w poszczególnych podgrupach. Implikuje to
fakt, iż podział jest zbalansowany i odpowiednio zróżnicowany dla k=2
ilości klastrów.

### Etap 4 - algorytm K-Średnich

Bliźniaczym algorytmem klasyfikacyjnym uczenia nienadzorowanego jest
algorytm **K-Means**. Zostanie on wykonany w celu porównania wyników obu
metod i wybrania odpowiedniej ilości klastrów.

``` r
ratio_ss <- rep(0, 7)
for (k in 1:7) {
 models <- kmeans(train_std[, 1:5], k, nstart = 20)
 ratio_ss[k] <- models$tot.withinss / models$totss

}
```

    ## Warning: 'medpolish()' nie zbiegł się w 10 iteracjach

``` r
plot(ratio_ss, type = "b", xlab = "k", main='Elbow plot')
```

![](projekt_files/figure-gfm/unnamed-chunk-25-1.png)<!-- -->

Powyżej przedstawiono tzw. *wykres złamanego łokcia* - kryterium
*złamanego kija*. Przedstawia on łamaną - *kij*, który załamuje się w
pewnych momentach. Jego analiza jest pomocna przy wyborze odpowiedniej
ilości klastrów dla metody **K-Means**. W tym przypadku, największe
załamanie następuje w k=2, lecz widoczne jest ono również dla k=4
(bardzo małe, lecz być może kluczowe). Dla dalszych k wykres staje się
liniowy, co oznacza, że dla wartości k większych od 4 nie jest możliwe
wyodrębnienie kolejncyh różnic między powstałymi 4 klastrami.

``` r
k2_means<-kmeans(train_std[, 1:5] ,centers=2, nstart = 20)
k4_means<-kmeans(train_std[, 1:5] ,centers=4, nstart = 20)
```

Powyżej wykonano algorytmy **k-średnich** dla wyznaczonej *metodą
łokcia* wartości k (2 oraz 4).

``` r
table(k2_means$cluster)
```

    ## 
    ##    1    2 
    ## 3520 3480

Powyżej pokazano liczebność danych w poszczególnych klastrach dla k=2.

``` r
table(k4_means$cluster)
```

    ## 
    ##    1    2    3    4 
    ## 1748 1750 1772 1730

Powyżej wyświetlono liczebność rekordów zbioru treningowego w
poszczególnych klastrach dla k=4. Podziały są bardzo zbalansowane, co
implikuje założenie, iż niemalże równoliczne klastry powstałe za pomocą
algorytmu **K-Średnich** dla k=2 miały bardziej subtelną strukturę i ich
dodatkowy podział prawdopodobnie (k=4) wydobył kolejne różnice w
poszczególnych grupach.

``` r
dunn(clusters=k2_means$cluster,Data=train_std[, 1:5])
```

    ## [1] 0.3169535

``` r
dunn(clusters=k4_means$cluster,Data=train_std[, 1:5])
```

    ## [1] 0.0104331

Powyżej wyznaczono *indeksy Dunna* dla wyników algorytmów **k-średnich**
dla k=2 i k=4. Wyniki znacznie różnią się od siebie, ponieważ miara
wspomnianego indeksu dla k=2 jest aż o rząd wyższa niż dla k=4. Oznacza
to, że największe załamanie na wykonanym wykresie *złamanego łokcia*
było najtrafniejszym odzwierciedleniem optymalnej ilości klastrów
zadanej dla algorytmu **K-means**, a pozorny równoliczny podział danych
na k=4 klastry był zatracaniem informacji i niepotrzebnym dzieleniem
zbioru.

### Etap 5 - wniosek końcowy dot. metod klasyfikacji

Zarówno metoda **aglomeracyjna hierarchicznej analizy skupień** dla
metryki **manhattan** z metodą **ward.D** jak i **k-means** wskazują na
jednoznaczne rozwiązanie - oba są odpowiednie do zastosowania w badanym
zbiorze danych. Wskaźnik Dunna dla **k-means** (k=2) wyniósł ok. 0.317,
co stanowi znaczącą poprawę w stosunku do drugiej konfiguracji (k=4:
0.010). Świadczy to o istnieniu dwóch bardzo wyraźnych profili studentów
w zbiorze treningowym. To sugeruje, iż oba wykonane algorytmy
klasyfikacyjne jednoznacznie wskazują na optymalny dobór wartości k,
równą 2. Badany zbiór ma jeden, bardzo wyraźny punkt podziału, którą
**k-means** (szukający centroidów) oraz **hierarchiczna analiza skupień
dla metody Warda** i **metryki Manhattan** (badająca struktrę danych)
bardzo dobrze zidentyfikowały.

### Etap 6 - algorytm kNN - k-najbliższych sąsiadów

``` r
train_labs <- as.factor(k2_means$cluster)
train_raw$klaster <- train_labs
```

Powyżej stworzono wektor wartości z nr klastrów na podstawie wybranego
algorytmu klasyfikacyjnego (**k-means** dla k=2). Został on dodany do
zbioru danych treningowych jako dodatkowa kolumna.

``` r
library(clue)
```

    ## Warning: pakiet 'clue' został zbudowany w wersji R 4.4.3

``` r
val_labs <- as.factor(cl_predict(k2_means, val_std[, 1:5]))
```

Powyżej z pomocą biblioteki **clue** dodano nr przyporządkowanych
klastrów do danych ze zbioru walidacyjnego z użyciem tego samego modelu,
co do zbioru treningowego w poprzednich etapach projektu. Dzięki temu
nie wykonano algorytmu **K-means** na podstawie zbioru walidacyjnego,
lecz dodano etykiety stosując ten sam model, co w przypadku zbioru
treningowego, aby sposób przypisywania był prawidłowy.

``` r
library(class)

k_vals <- 2:10
accuracy <- sapply(k_vals, function(k_val) {
  pred <- knn(train = train_std[, 1:5], 
              test = val_std[, 1:5], 
              cl = train_labs, 
              k = k_val)
  mean(pred == val_labs)
})

accuracy
```

    ## [1] 1 1 1 1 1 1 1 1 1

Powyżej wykonano modele **kNN** trenując model na zbiorze treningowym i
testując na zbiorze walidacyjnym dla wartości k sąsiadów od 2 do 10.
Metryką decydującą o jakości modelu jest **accuracy**. Jak pokazano, dla
każdej wartości k accuracy wynosi 1.0, co oznacza, że algorytm
**k-means** perfekcyjnie odseparował klastry a struktura danych jest
niezwykle wyraźna. Dlatego też jako optymalną liczbę sąsiadów k można
wybrać dowolną liczbę z analizowanego przedziału (np. k=3).

``` r
final_pred_4 <- knn(train = train_std[, 1:5], 
                    test = test_std[, 1:5], 
                    cl = train_labs, 
                    k = 3)

final_score <- data.frame(
  Obserwacja = 1:4,
  nr_klastra = final_pred_4,
  Performance.Index = test_raw$Performance.Index
)

print(final_score)
```

    ##   Obserwacja nr_klastra Performance.Index
    ## 1          1          2                43
    ## 2          2          2                78
    ## 3          3          1                27
    ## 4          4          2                70

Powyższe zestawienie nr klastrów, do którego zostali przyporządkowani
studenci (zbiór testowy - 4 obserwacje) z indeksami wydajności, jakie są
do nich przypisane prowadzi do interesującej obserwacji. Studenci
wykazujący się zarówno wysokim jak i przeciętnym współczynnikiem
wydajności zostali przyporządkowani do tego samego klastra. Dopiero
osoba z niższą wartością *Performance.Index* została przypisana do 1
klastra. Poniżej została zestawiona średnia wartość każdej zmiennej w
zbiorze treningowym dla osób w poszczególnych klastrach.

``` r
aggregate(cbind(Hours.Studied, Extracurricular.Activities, Previous.Scores, Sleep.Hours, Sample.Question.Papers.Practiced, Performance.Index) ~ train_labs, 
          data = train_raw, 
          FUN = mean)
```

    ##   train_labs Hours.Studied Extracurricular.Activities Previous.Scores
    ## 1          1      4.952557                          0        69.31875
    ## 2          2      4.985920                          1        69.39713
    ##   Sleep.Hours Sample.Question.Papers.Practiced Performance.Index
    ## 1    6.557670                         4.515057          54.68097
    ## 2    6.493391                         4.565805          55.43477

### Etap 7 - wnioski końcowe projektu

#### Przeprowadzona analiza doprowadziła do sformułowania nieoczywistych wniosków dotyczących zbioru Student Performance. Zastosowane algorytmy k-średnich oraz k-najbliższych sąsiadów wykazały wyjątkową spójność matematyczną:

- Indeks Dunn na poziomie 0.32 świadczy o wyraźnej separacji klastrów w
  przestrzeni cech,  
- Stuprocentowa dokładność (Accuracy) modelu kNN (1.0) potwierdza, że
  granice między grupami są ostre i jednoznaczne dla algorytmu.

#### Mimo tej *perfekcji* matematycznej, analiza profilowania ujawniła paradoks decyzyjny: średni wskaźnik wydajności (*Performance Index*) dla obu grup jest niemal identyczny (54.7 vs 55.4). Szczegółowe zestawienie średnich pozwoliło zidentyfikować główną przyczynę tego podziału – algorytmy dokonały segregacji studentów niemal wyłącznie na podstawie zmiennej binarnej *Extracurricular Activities*. Klaster 1 grupuje wyłącznie osoby niebiorące udziału w zajęciach dodatkowych, natomiast Klaster 2 to studenci aktywni pozalekcyjnie. Granica ta, ze względu na naturę tej zmiennej, jest na tyle oczywista, że model osiągnął bardzo dobre rezultaty (wspomniane wyżej). Jednakże, ze względu na podobne średnie wartości wskaźnika wydajności w obu klastrach, udział w zajęciach dodatkowych (który tak silnie różnicuje styl życia studentów) nie przekłada się na lepsze wyniki w nauce w tym zbiorze danych.

#### Ostatecznie, projekt udowodnił skuteczność metod klasyfikacji w rozpoznawaniu wzorców zachowań, jednocześnie obalając prostą zależność między badanymi nawykami a efektywnością nauki w tej konkretnej populacji.
