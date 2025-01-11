<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class UsageLog extends Model
{
    use HasFactory;

    protected $fillable = ['toilet_id', 'usage_timestamp'];

    // Each usage log is associated with a specific toilet
    public function toilet()
    {
        return $this->belongsTo(Toilet::class);
    }
}
