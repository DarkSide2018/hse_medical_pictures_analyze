
CREATE SCHEMA "analyze_medical";

CREATE USER hse_medical_user WITH PASSWORD '123456';


CREATE EXTENSION IF NOT EXISTS "uuid-ossp";



create table "analyze_medical".medical_pictures_train(
                              picture_id              UUID        NOT NULL PRIMARY KEY DEFAULT uuid_generate_v4(),
                              created_at             TIMESTAMP    NOT NULL DEFAULT now(),
                              updated_at             TIMESTAMP    NULL,
                              target                 integer not null,
                              Label  TEXT NOT null,
                              image_path  TEXT NOT null,
                              red_channel_intensity numeric(10,5) not null,
                              blue_channel_intensity numeric(10,5) not null,
                              green_channel_intensity numeric(10,5) not null,
                              HOG_mean numeric(10,5) not null,
                              harris_count integer null,
                              harris_count_mean numeric(10, 5) NULL,
                              HOG_std numeric(10,5) not null
);


create table "analyze_medical".medical_pictures_test(
                              picture_id              UUID        NOT NULL PRIMARY KEY DEFAULT uuid_generate_v4(),
                              created_at             TIMESTAMP    NOT NULL DEFAULT now(),
                              updated_at             TIMESTAMP    NULL,
                              target                 integer not null,
                              Label  TEXT NOT null,
                              image_path  TEXT NOT null,
                              red_channel_intensity numeric(10,5) not null,
                              blue_channel_intensity numeric(10,5) not null,
                              green_channel_intensity numeric(10,5) not null,
                              HOG_mean  numeric(10,5) not null,
                              harris_count integer null,
                              harris_count_mean numeric(10, 5) NULL,
                              HOG_std  numeric(10,5) not null
);

create table "analyze_medical".target_dictionary(
                              id              UUID        NOT NULL PRIMARY KEY DEFAULT uuid_generate_v4(),
                              created_at             TIMESTAMP    NOT NULL DEFAULT now(),
                              updated_at             TIMESTAMP    NULL,
                              target                 integer not null,
                              Label  TEXT NOT null
);


INSERT INTO "analyze_medical".target_dictionary (target,"label") VALUES
	 (0,'Eczema Photos'),
	 (1,'Acne and Rosacea Photos'),
	 (2,'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions'),
	 (3,'Psoriasis pictures Lichen Planus and related diseases'),
	 (4,'Tinea Ringworm Candidiasis and other Fungal Infections'),
	 (5,'Lupus and other Connective Tissue diseases'),
	 (6,'Light Diseases and Disorders of Pigmentation'),
	 (7,'Melanoma Skin Cancer Nevi and Moles'),
	 (8,'Nail Fungus and other Nail Disease'),
	 (9,'Atopic Dermatitis Photos'),
	 (10,'Bullous Disease Photos'),
	 (11,'Cellulitis Impetigo and other Bacterial Infections'),
	 (12,'Exanthems and Drug Eruptions'),
	 (13,'Hair Loss Photos Alopecia and other Hair Diseases'),
	 (14,'Herpes HPV and other STDs Photos'),
	 (15,'Poison Ivy Photos and other Contact Dermatitis'),
	 (16,'Scabies Lyme Disease and other Infestations and Bites'),
	 (17,'Seborrheic Keratoses and other Benign Tumors'),
	 (18,'Systemic Disease'),
	 (19,'Urticaria Hives'),
	 (20,'Vascular Tumors'),
	 (21,'Vasculitis Photos'),
	 (22,'Warts Molluscum and other Viral Infections');


ALTER TABLE "analyze_medical".target_dictionary ADD CONSTRAINT label_unique UNIQUE (Label);

CREATE INDEX image_path_index_train ON "analyze_medical".medical_pictures_train(image_path);
CREATE INDEX image_path_index_test ON "analyze_medical".medical_pictures_test(image_path);

create table "analyze_medical".med_pictures_pca_train(
                              picture_id              UUID        NOT NULL PRIMARY KEY DEFAULT uuid_generate_v4(),
                              created_at             TIMESTAMP    NOT NULL DEFAULT now(),
                              updated_at             TIMESTAMP    NULL,
                              target                 integer not null,
                              Label  TEXT NOT null,
                              image_path  TEXT NOT null,
                              red_channel_intensity numeric(10,5) not null,
                              blue_channel_intensity numeric(10,5) not null,
                              green_channel_intensity numeric(10,5) not null,
                              HOG_mean numeric(10,5) not null,
                              harris_count integer null,
                              harris_count_mean numeric(10, 5) NULL,
                              HOG_std numeric(10,5) not null
);
create table "analyze_medical".med_pictures_pca_test(
                              picture_id              UUID        NOT NULL PRIMARY KEY DEFAULT uuid_generate_v4(),
                              created_at             TIMESTAMP    NOT NULL DEFAULT now(),
                              updated_at             TIMESTAMP    NULL,
                              target                 integer not null,
                              Label  TEXT NOT null,
                              image_path  TEXT NOT null,
                              red_channel_intensity numeric(10,5) not null,
                              blue_channel_intensity numeric(10,5) not null,
                              green_channel_intensity numeric(10,5) not null,
                              HOG_mean numeric(10,5) not null,
                              harris_count integer null,
                              harris_count_mean numeric(10, 5) NULL,
                              HOG_std numeric(10,5) not null
);

CREATE INDEX image_path_index_train ON "analyze_medical".med_pictures_pca_train(image_path);
CREATE INDEX image_path_index_test ON "analyze_medical".med_pictures_pca_test(image_path);