
# FlexDetect Machine Learning Service

## Vsebina
- [Namen mikrostoritve](#namen-mikrostoritve)
- [Modeli in algoritmi](#modeli-in-algoritmi)
- [Arhitektura](#arhitektura)
- [Vhodni in izhodni podatki](#vhodni-in-izhodni-podatki)
- [API specifikacija](#api-specifikacija)
- [Primeri zahtevkov](#primeri-zahtevkov)
- [Integracija](#integracija)
- [Razvojne smernice](#razvojne-smernice)

---

## Namen mikrostoritve
Obdelava in analiza uvoženih podatkov z uporabo naprednih modelov strojnega učenja za zaznavo dogodkov odziva na povpraševanje in oceno njihovega vpliva na porabo energije.

---

## Modeli in algoritmi
- Metode nadziranega učenja: Random Forest, Gradient Boosting
- Prilagojeni algoritmi za detekcijo anomalij
- Avtomatska optimizacija hiperparametrov

---

## Arhitektura
- **Jezik:** Python 3.10
- **Okvir:** TensorFlow, scikit-learn, PyTorch
- **Okolje:** Google Cloud Functions za skalabilno izvajanje
- **Podpora:** REST API za prejem podatkov in vračanje rezultatov

---

## Vhodni in izhodni podatki
- **Vhod:** Časovne vrste očiščenih podatkov in metapodatki
- **Izhod:** Identifikacija dogodkov, trajanje, magnitude odziva, verjetnosti

---

## API specifikacija

| Endpoint             | Metoda | Opis                                 |
|----------------------|--------|-------------------------------------|
| `/ml/run`            | POST   | Zažene analizo in napoved            |
| `/ml/status/{id}`    | GET    | Pridobi status izvajanja modela      |
| `/ml/results/{id}`   | GET    | Pridobi rezultate analize             |

---

## Primeri zahtevkov

```json
{
  "dataId": "67890",
  "parameters": {
    "model": "random_forest",
    "threshold": 0.7
  }
}
```

---

## Integracija
- Prejem očiščenih podatkov iz **flexdetect-data-service**
- Posredovanje rezultatov mikrostoritvi **flexdetect-visualization-service**
- Uporabniški dostop preko **flexdetect-user-service**
---

## Razvojne smernice
- Redno treniranje modelov z novimi podatki
- Spremljanje metrike učinkovitosti in natančnosti modelov
---

**Avtor:** Aljaž Brodar  
**Zadnja posodobitev:** 1. december 2025
