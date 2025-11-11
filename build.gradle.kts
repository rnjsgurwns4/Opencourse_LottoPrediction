val ktorVersion = "2.3.11"
val dataframeVersion = "0.12.0"
val wekaVersion = "3.8.6"
plugins {
    kotlin("jvm") version "2.1.10"
    application
    id("org.jetbrains.kotlin.plugin.serialization") version "2.1.10"
}

group = "org.example"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    implementation("io.ktor:ktor-client-cio:$ktorVersion") // The engine
    implementation("io.ktor:ktor-client-content-negotiation:$ktorVersion") // For JSON
    implementation("io.ktor:ktor-serialization-kotlinx-json:$ktorVersion") // The JSON parser

    // 2. Kotlinx DataFrame: For manipulating data (like Python Pandas)
    implementation("org.jetbrains.kotlinx:dataframe:$dataframeVersion")

    // 3. Weka: The Machine Learning library (Java-based, works perfectly with Kotlin)
    implementation("nz.ac.waikato.cms.weka:weka-stable:$wekaVersion")

    // Optional: Add a logging library for Ktor (helps with debugging)
    implementation("ch.qos.logback:logback-classic:1.5.6")

}

tasks.test {
    useJUnitPlatform()
}
kotlin {
    jvmToolchain(21)
}

application {
    mainClass.set("lotto.MainKt")
}


tasks.withType<Test> {
    useJUnitPlatform()
}