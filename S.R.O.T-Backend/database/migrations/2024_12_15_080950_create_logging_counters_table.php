<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        Schema::create('logging_counters', function (Blueprint $table) {
            $table->id();
            $table->unsignedBigInteger('toilet_id');
            $table->unsignedBigInteger('floor_id');
            $table->unsignedBigInteger('building_id');
            $table->integer('usage_count');
            $table->unsignedBigInteger('gender_id')->nullable();
            $table->boolean('is_reset')->default(false);
            $table->timestamp('logged_at')->useCurrent();
            $table->timestamps();

            $table->foreign('toilet_id')->references('id')->on('toilets')->onDelete('cascade');
            $table->foreign('floor_id')->references('id')->on('floors')->onDelete('cascade');
            $table->foreign('building_id')->references('id')->on('buildings')->onDelete('cascade');
            $table->foreign('gender_id')->references('id')->on('genders')->onDelete('cascade');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('logging_counters');
    }
};
