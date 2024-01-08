
CREATE SCHEMA "analyze";

CREATE USER hse_medical_user WITH PASSWORD '123456';


CREATE EXTENSION IF NOT EXISTS "uuid-ossp";



create table "analyze".medical_pictures_train(
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
                              HOG_std numeric(10,5) not null
);


create table "analyze".medical_pictures_test(
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
                              HOG_std  numeric(10,5) not null
);

create table "analyze".target_dictionary(
                              id              UUID        NOT NULL PRIMARY KEY DEFAULT uuid_generate_v4(),
                              created_at             TIMESTAMP    NOT NULL DEFAULT now(),
                              updated_at             TIMESTAMP    NULL,
                              target                 integer not null,
                              Label  TEXT NOT null
);


INSERT INTO "analyze".target_dictionary (
                           target,
                           Label)
                           VALUES
                           (0,'Eczema Photos'),
                           (1,'Acne and Rosacea Photos'),
                           (2,'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions'),
                           (3,'Psoriasis pictures Lichen Planus and related diseases'),
                           (5,'Lupus and other Connective Tissue diseases'),
                           (6,'Light Diseases and Disorders of Pigmentation'),
                           (7,'Melanoma Skin Cancer Nevi and Moles'),
                           (8,'Nail Fungus and other Nail Disease'),
                           (4,'Tinea Ringworm Candidiasis and other Fungal Infections');


ALTER TABLE "analyze".target_dictionary ADD CONSTRAINT label_unique UNIQUE (Label);

CREATE INDEX image_path_index_train ON "analyze".medical_pictures_train(image_path);
CREATE INDEX image_path_index_test ON "analyze".medical_pictures_test(image_path);