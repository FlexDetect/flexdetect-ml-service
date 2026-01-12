# FlexDetect Machine Learning Service

## Vsebina
- Namen mikrostoritve
- Funkcionalnost in logika
- Arhitektura
- Vhodni in izhodni podatki
- API specifikacija
- Primer zahtevka
- Integracija z drugimi mikrostoritvami
- Razvojne opombe

---

## Namen mikrostoritve

Mikrostoritev **FlexDetect ML Service** izvaja zaznavo dogodkov energetske prilagodljivosti (Demand Response – DR) na podlagi časovnih vrst meritev porabe energije.

Storitev:
- pridobi meritve iz **data-service**,
- izvede analizo časovne vrste,
- zazna potencialne DR dogodke,
- izračuna osnovne povzetke,
- rezultate vrne prek REST API-ja.

---

## Funkcionalnost in logika

Logika zaznavanja temelji na:
- osnovni obdelavi časovne vrste,
- izračunu baseline porabe,
- primerjavi dejanske porabe z baseline,
- zaznavi intervalov z večjim odstopanjem.

Model ni učeč (brez treniranja); uporablja deterministične metode, implementirane v `model.py`.

---

## Arhitektura

- Jezik: Python 3  
- API okvir: FastAPI  
- Obdelava podatkov: pandas  
- Komunikacija: REST (JSON)  
- Zagon: Docker / Azure Container Apps  

Storitev je stateless.

---

## Vhodni in izhodni podatki

### Vhod
Storitev sprejme ID-je podatkovnih nizov in meritev, nato podatke pridobi iz data-service.

### Izhod
- zaznani DR dogodki,
- časovna vrsta z oznakami dogodkov,
- osnovni povzetek zaznave.

---

## API specifikacija

### POST /detect

```json
{
  "dataset_id": 1,
  "power_measurement_id": 10,
  "feature_measurement_ids": [11, 12]
}
```

---

## Integracija z drugimi mikrostoritvami

- Data Service: pridobivanje meritev
- Frontend: prikaz rezultatov
- User Service: JWT avtentikacija

---

## Razvojne opombe

- Brez persistentnega stanja
- Nastavljiv timeout za klic data-service
- Namenjeno demonstraciji zaznave DR dogodkov

---

Avtor: Aljaž Brodar  
Zadnja posodobitev: 12. januar 2026
